"""Prepare a read-only episode allowlist for the label-down fast run.

The source LeRobot dataset is never rewritten.  The prepared directory links
the original data/video/meta tables and contains only an annotation allowlist;
VLADataset then asks LeRobotDataset to select those original episode indices.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


SRC = Path("/home/bjtc/Sophix/datasets/marked_data_0916")
OUT = Path("/home/bjtc/Sophix/datasets/label_down_fast_150")
SEED = 20260921
N = 150


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def main() -> None:
    labels_path = SRC / "meta/annotations/episode_labels.jsonl"
    labels = {}
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            labels[int(rec["episode_index"])] = rec

    ep_table = pq.read_table(SRC / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    meta = {int(ep_table["episode_index"][i]): {k: ep_table[k][i] for k in ep_table}
            for i in range(len(ep_table["episode_index"]))}

    # “Good” is an auditable data-quality rule, not the business category:
    # approved/succeeded, no regrasp, label initially down, finite source
    # actions, and a normal-length episode.  The category is intentionally not
    # used because label_down is recorded as category=failure by the annotation
    # schema even when the demonstration is valid for this conditional task.
    candidates = []
    for ep, rec in labels.items():
        ann = rec.get("extra_labels", {}).get("self_annotation", {})
        if ann.get("grasp_label_up") is not False or int(ann.get("had_regrasp", 1)) != 0:
            continue
        if rec.get("status") != "approved" or rec.get("execution_status") != "succeeded":
            continue
        if rec.get("labels", {}).get("failure_reasons") not in ([], None):
            continue
        m = meta[ep]
        length = int(m["length"])
        if not 500 <= length <= 800:
            continue
        p = SRC / f"data/chunk-{int(m['data/chunk_index']):03d}/file-{int(m['data/file_index']):03d}.parquet"
        arr = np.asarray(pq.read_table(p, columns=["action"]).column("action").to_pylist(), dtype=np.float64)
        finite = bool(np.isfinite(arr).all())
        if not finite:
            continue
        step_deg = np.rad2deg(np.abs(np.diff(arr[:, :14], axis=0)))
        candidates.append({
            "episode_index": ep,
            "length": length,
            "p99_step_deg": float(np.quantile(step_deg, 0.99)),
            "p999_step_deg": float(np.quantile(step_deg, 0.999)),
            "max_step_deg": float(step_deg.max(initial=0.0)),
            "mean_step_deg": float(step_deg.mean()),
            "source_data_file": str(p),
            "dataset_from_index": int(m["dataset_from_index"]),
            "dataset_to_index": int(m["dataset_to_index"]),
            "finite_action": finite,
        })

    if len(candidates) < N:
        raise RuntimeError(f"Only {len(candidates)} good label-down candidates; need {N}")

    # Preserve length diversity while preferring smooth trajectories.  Ten
    # length strata x fifteen episodes gives a reproducible, non-biased subset.
    candidates.sort(key=lambda x: (x["length"], x["episode_index"]))
    rng = np.random.default_rng(SEED)
    selected = []
    for bucket in np.array_split(np.asarray(candidates, dtype=object), 10):
        bucket = list(bucket)
        bucket.sort(key=lambda x: (x["max_step_deg"], x["p999_step_deg"], x["episode_index"]))
        # A deterministic jitter only resolves near-ties; quality order remains
        # primary and episode indices stay original.
        take = bucket[:15]
        selected.extend(take)
    selected = sorted(selected, key=lambda x: x["episode_index"])
    assert len(selected) == N and len({x["episode_index"] for x in selected}) == N

    # Build a lightweight view.  No source table is copied or rewritten.
    OUT.mkdir(parents=True, exist_ok=True)
    link_or_copy(SRC / "data", OUT / "data")
    link_or_copy(SRC / "videos", OUT / "videos")
    (OUT / "meta").mkdir(exist_ok=True)
    for item in (SRC / "meta").iterdir():
        if item.name == "annotations":
            continue
        link_or_copy(item, OUT / "meta" / item.name)
    ann_out = OUT / "meta/annotations"
    ann_out.mkdir(exist_ok=True)
    ann_out.joinpath("episode_labels.jsonl").write_text(
        "".join(json.dumps(labels[x["episode_index"]], ensure_ascii=False) + "\n" for x in selected),
        encoding="utf-8",
    )

    report = {
        "source": str(SRC),
        "output_view": str(OUT),
        "selection_seed": SEED,
        "selection_rule": "label_down, had_regrasp=0, status=approved, execution_status=succeeded, failure_reasons empty, length 500..800, 10 length strata x 15 smoothest",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_episode_indices": [x["episode_index"] for x in selected],
        "selected_frame_count": int(sum(x["length"] for x in selected)),
        "selected_metrics": selected,
        "source_info_sha256": hashlib.sha256((SRC / "meta/info.json").read_bytes()).hexdigest(),
        "source_labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "mapping_invariant": "original episode_index and dataset_from_index/dataset_to_index retained; data/video/meta tables are symlinked to source",
    }
    (OUT / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate_count", "selected_count", "selected_frame_count", "output_view")}, indent=2))


if __name__ == "__main__":
    main()
