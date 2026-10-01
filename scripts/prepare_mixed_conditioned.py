"""Merge separated label datasets while preserving episode labels for prompt conditioning."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/bjtc/Sophix")
SRC = ROOT / "datasets/separated_data"
DST = ROOT / "datasets/mixed_conditioned_data"
HORIZON = 50


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def hardlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.stat().st_ino != dst.stat().st_ino:
            raise FileExistsError(dst)
        return
    os.link(src, dst)


def digest(path: Path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def stats(values):
    values = np.asarray(values)
    q = np.quantile(values, [.01, .99, .02, .98], axis=0)
    return {
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.astype(np.float64).std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": q[0].tolist(), "q99": q[1].tolist(),
        "q02": q[2].tolist(), "q98": q[3].tolist(),
    }


def video_keys(info):
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def build_split(split: str):
    sources = [SRC / "label_up" / split, SRC / "label_down" / split]
    out = DST / split
    out.mkdir(parents=True)
    for rel in ("data/chunk-000", "meta/episodes/chunk-000", "meta/annotations", "meta/alignments"):
        (out / rel).mkdir(parents=True, exist_ok=True)

    first_info = json.loads((sources[0] / "meta/info.json").read_text())
    vkeys = video_keys(first_info)
    rows = []
    annotations = []
    provenance = []
    mapping = []
    checksums = []
    state_values, action_values, timestamp_values = [], [], []
    offset = 0

    for source in sources:
        source_info = json.loads((source / "meta/info.json").read_text())
        source_eps = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        source_labels = {int(x["episode_index"]): x for x in read_jsonl(source / "meta/annotations/episode_labels.jsonl")}
        source_prov = {int(x["episode_index"]): x for x in read_jsonl(source / "meta/episodes.jsonl")}

        for old_ep in range(len(source_eps)):
            ep = copy.deepcopy(source_eps[old_ep])
            new_ep = len(rows)
            data_src = source / "data/chunk-000" / f"file-{old_ep:03d}.parquet"
            table = pq.read_table(data_src)
            n = len(table)
            table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", pa.array(np.full(n, new_ep, dtype=np.int64)))
            table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(np.arange(offset, offset + n, dtype=np.int64)))
            data_dst = out / "data/chunk-000" / f"file-{new_ep:03d}.parquet"
            pq.write_table(table, data_dst, compression="zstd")
            checksums.append({"source": str(data_src), "target": str(data_dst.relative_to(out)), "target_sha256": digest(data_dst)})

            ep["episode_index"] = new_ep
            ep["data/chunk_index"] = 0
            ep["data/file_index"] = new_ep
            ep["dataset_from_index"] = offset
            ep["dataset_to_index"] = offset + n
            for key in vkeys:
                prefix = f"videos/{key}"
                ep[f"{prefix}/chunk_index"] = 0
                ep[f"{prefix}/file_index"] = new_ep
                src_video = source / prefix / "chunk-000" / f"file-{old_ep:03d}.mp4"
                dst_video = out / prefix / "chunk-000" / f"file-{new_ep:03d}.mp4"
                hardlink(src_video, dst_video)
                ep[f"{prefix}/from_timestamp"] = 0.0
                ep[f"{prefix}/to_timestamp"] = (n - 1) / float(source_info.get("fps", 30))
            rows.append(ep)

            label = copy.deepcopy(source_labels[old_ep])
            label["episode_index"] = new_ep
            label["source_split"] = split
            label["source_group"] = source.parent.name
            label["alignment_path"] = f"meta/alignments/episode-{new_ep:06d}.json"
            annotations.append(label)
            prov = copy.deepcopy(source_prov[old_ep])
            prov["episode_index"] = new_ep
            prov["alignment_path"] = f"meta/alignments/episode-{new_ep:06d}.json"
            prov["source_split"] = split
            prov["source_group"] = source.parent.name
            provenance.append(prov)

            src_align = source / source_prov[old_ep]["alignment_path"]
            dst_align = out / "meta/alignments" / f"episode-{new_ep:06d}.json"
            hardlink(src_align, dst_align)
            mapping.append({
                "episode_index": new_ep,
                "source_group": source.parent.name,
                "source_split": split,
                "source_episode_index": old_ep,
                "grasp_label_up": bool(label["extra_labels"]["self_annotation"]["grasp_label_up"]),
                "had_regrasp": int(label["extra_labels"]["self_annotation"]["had_regrasp"]),
                "frames": n,
            })
            state_values.append(np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 16))
            action_values.append(np.asarray(table["action"].combine_chunks().values).reshape(-1, 16))
            timestamp_values.append(np.asarray(table["timestamp"]))
            offset += n

    schema = pq.read_table(sources[0] / "meta/episodes/chunk-000/file-000.parquet").schema
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "meta/episodes/chunk-000/file-000.parquet", compression="zstd")
    write_jsonl(out / "meta/annotations/episode_labels.jsonl", annotations)
    write_jsonl(out / "meta/episodes.jsonl", provenance)
    write_jsonl(out / "episode_mapping.jsonl", mapping)
    write_jsonl(out / "file_checksums.jsonl", checksums)
    shutil.copy2(sources[0] / "meta/tasks.parquet", out / "meta/tasks.parquet")

    info = copy.deepcopy(first_info)
    info.update(total_episodes=len(rows), total_frames=offset, splits={split: f"0:{len(rows)}"}, source_dataset_root=str(SRC))
    write_json(out / "meta/info.json", info)
    all_state = np.concatenate(state_values)
    all_action = np.concatenate(action_values)
    all_time = np.concatenate(timestamp_values)[:, None]
    write_json(out / "meta/stats.json", {
        "observation.state": stats(all_state), "action": stats(all_action), "timestamp": stats(all_time)
    })
    return mapping, all_state, all_action


def compute_norm(train_root: Path, state: np.ndarray, action: np.ndarray, mapping):
    norm = {
        "observation.state.arm.position": stats(state[:, :14]),
        "observation.state.effector.position": stats(state[:, 14:]),
        "action.effector.position": stats(action[:, 14:]),
    }
    # Episode-tail hold, same definition as separated-data norm generation.
    arm_parts = []
    for row in mapping:
        ep = int(row["episode_index"])
        table = pq.read_table(train_root / "data/chunk-000" / f"file-{ep:03d}.parquet")
        s = np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 16)
        a = np.asarray(table["action"].combine_chunks().values).reshape(-1, 16)
        for h in range(HORIZON):
            idx = np.minimum(np.arange(len(a)) + h, len(a) - 1)
            arm_parts.append(a[idx, :14] - s[:, :14])
    arm = {}
    arm_values = np.concatenate(arm_parts)
    for key, value in stats(arm_values).items():
        arm[key] = [value]
    # Expand per-horizon statistics, matching existing norm file contract.
    arm = {}
    for h in range(HORIZON):
        vals = []
        for row in mapping:
            table = pq.read_table(train_root / "data/chunk-000" / f"file-{int(row['episode_index']):03d}.parquet")
            s = np.asarray(table["observation.state"].combine_chunks().values).reshape(-1, 16)
            a = np.asarray(table["action"].combine_chunks().values).reshape(-1, 16)
            idx = np.minimum(np.arange(len(a)) + h, len(a) - 1)
            vals.append(a[idx, :14] - s[:, :14])
        sh = stats(np.concatenate(vals))
        for key, value in sh.items():
            arm.setdefault(key, []).append(value)
    norm["action.arm.position"] = arm
    out = train_root.parent / "train_stats" / "norm_stats.json"
    write_json(out, {"norm_stats": norm, "count": int(len(state))})
    write_json(out.parent / "provenance.json", {"source_split": str(train_root), "train_episodes": len(mapping), "chunk_size": HORIZON, "method": "mixed train only; episode-conditioned prompts; arm action is future target minus current state with tail hold", "sha256": digest(out)})


def main():
    if DST.exists():
        raise FileExistsError(f"Refusing to overwrite {DST}")
    DST.mkdir(parents=True)
    train_mapping, train_state, train_action = build_split("train")
    test_mapping, _, _ = build_split("test")
    compute_norm(DST / "train", train_state, train_action, train_mapping)
    manifest = {
        "source": str(SRC), "destination": str(DST), "split_unit": "episode",
        "prompt_conditioning": "grasp_label_up episode annotation selects task prompt in VLADataset",
        "splits": {"train": {"episodes": len(train_mapping)}, "test": {"episodes": len(test_mapping)}},
        "label_counts": {
            "train_up": sum(int(x["grasp_label_up"]) for x in train_mapping),
            "train_down": sum(not x["grasp_label_up"] for x in train_mapping),
            "test_up": sum(int(x["grasp_label_up"]) for x in test_mapping),
            "test_down": sum(not x["grasp_label_up"] for x in test_mapping),
        },
        "hardlinked_videos": True,
    }
    write_json(DST / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
