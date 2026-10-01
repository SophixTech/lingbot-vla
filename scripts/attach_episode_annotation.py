#!/usr/bin/env python3
"""Attach a platform annotation to an immutable LeRobot conversion.

The annotation is metadata only: raw HDF5/images are never opened for writing,
and rejected or structurally unsafe episodes are explicitly marked unusable
for training.  Event frames are mapped only when they belong to a retained
continuous segment; no frame is fabricated across an action gap.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq


def load_json(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--annotation", type=Path, required=True)
    ap.add_argument("--raw-episode", type=Path, required=True)
    ap.add_argument("--converter-pyc", type=Path, required=True)
    ap.add_argument("--tolerance-ms", type=float, default=25.0)
    args = ap.parse_args()

    out = args.output.resolve()
    ann = load_json(args.annotation)
    report = load_json(out / "conversion_report.json")

    spec = importlib.util.spec_from_file_location("a2d_converter", args.converter_pyc)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load converter")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plan = mod.scan_episode(args.raw_episode, int(args.tolerance_ms * 1_000_000), 20)
    runs = [list(map(int, r)) for r in plan["runs"]]

    events = {}
    for name, value in ann.get("events", {}).items():
        source_frame = value.get("frame") if isinstance(value, dict) else value
        mapped = None
        if source_frame is not None:
            for out_ep, run in enumerate(runs):
                if int(source_frame) in run:
                    mapped = {"output_episode": out_ep,
                              "output_frame": run.index(int(source_frame)),
                              "source_frame": int(source_frame)}
                    break
        events[name] = {
            "source_frame": source_frame,
            "source_timestamp_ns": value.get("timestamp_ns") if isinstance(value, dict) else None,
            "mapped": mapped,
        }

    reasons = []
    if ann.get("status") != "approved":
        reasons.append(f"annotation_status_{ann.get('status', 'missing')}")
    if len(runs) != 1:
        reasons.append("source_action_timeline_split_into_multiple_segments")
    unmapped = [k for k, v in events.items() if v["source_frame"] is not None and v["mapped"] is None]
    if unmapped:
        reasons.append("event_frames_removed_by_action_gap:" + ",".join(sorted(unmapped)))

    # Validate the generated tabular payload without changing it.
    parquet = next((out / "data").glob("**/*.parquet"))
    table = pq.read_table(parquet)
    finite = True
    for col in ("observation.state", "action"):
        for row in table[col].to_pylist():
            finite = finite and all(float(x) == float(x) and abs(float(x)) != float("inf") for x in row)
    shape_ok = all(len(row) == 16 for row in table["observation.state"].to_pylist()) and all(
        len(row) == 16 for row in table["action"].to_pylist())
    if not finite:
        reasons.append("nan_or_inf_in_state_or_action")
    if not shape_ok:
        reasons.append("state_or_action_shape_not_16")

    labels_dir = out / "meta" / "annotations"
    labels_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": ann.get("schema_version"),
        "source_episode_id": ann.get("source_episode_id"),
        "platform_episode_id": ann.get("episode_id"),
        "annotation_id": ann.get("annotation_id"),
        "annotation_revision": ann.get("revision"),
        "annotation_status": ann.get("status"),
        "labels": ann.get("labels", {}),
        "reviews": ann.get("reviews", []),
        "events": events,
        "conversion_output_episodes": report.get("output_episodes"),
        "conversion_output_frames": report.get("output_frames"),
        "training_eligible": not reasons,
        "training_blockers": reasons,
    }
    (labels_dir / "episode_labels.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")

    audit = {
        "source_episode_id": ann.get("source_episode_id"),
        "output": str(out),
        "lerobot_v3_structure": True,
        "table_rows": table.num_rows,
        "state_action_shape_16": shape_ok,
        "finite_state_action": finite,
        "camera_keys": [k for k in load_json(out / "meta" / "info.json")["features"] if k.startswith("observation.images.")],
        "annotation_status": ann.get("status"),
        "training_eligible": not reasons,
        "blockers": reasons,
        "note": "Rejected/unsafe records remain available for audit; raw source is unchanged.",
    }
    (out / "compatibility_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"annotation_file": str(labels_dir / "episode_labels.jsonl"), "audit_file": str(out / "compatibility_audit.json"), "training_eligible": not reasons, "blockers": reasons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
