"""Build a balanced 75/75 training dataset from marked_data_0916.

The source dataset is read-only. Episode parquet files are rewritten with
contiguous local indices, while videos and alignment files are hard-linked to
avoid duplicating the large media payload.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/home/bjtc/Sophix")
SOURCE = ROOT / "datasets/marked_data_0916"
DEST = ROOT / "datasets/mixed_few_data"
SEED = 20260930
PER_LABEL = 75
HORIZON = 50


def digest(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    q = np.quantile(values, [0.01, 0.99, 0.02, 0.98], axis=0)
    return {
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.astype(np.float64).std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": q[0].tolist(),
        "q99": q[1].tolist(),
        "q02": q[2].tolist(),
        "q98": q[3].tolist(),
    }


def vector(table: pa.Table, key: str) -> np.ndarray:
    return table[key].combine_chunks().values.to_numpy().reshape(-1, 16)


def hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def load_source():
    info = json.loads((SOURCE / "meta/info.json").read_text())
    labels = {
        int(row["episode_index"]): row
        for row in map(json.loads, (SOURCE / "meta/annotations/episode_labels.jsonl").read_text().splitlines())
        if row
    }
    provenance = {
        int(row["episode_index"]): row
        for row in map(json.loads, (SOURCE / "meta/episodes.jsonl").read_text().splitlines())
        if row
    }
    ep_table = pq.read_table(SOURCE / "meta/episodes/chunk-000/file-000.parquet")
    episodes = {int(row["episode_index"]): row for row in ep_table.to_pylist()}
    expected = set(range(int(info["total_episodes"])))
    assert set(labels) == set(provenance) == set(episodes) == expected
    return info, labels, provenance, episodes, ep_table.schema


def select_episodes(labels, episodes):
    groups = {True: [], False: []}
    for episode_index, label in labels.items():
        value = label["extra_labels"]["self_annotation"]["grasp_label_up"]
        assert type(value) is bool
        meta = episodes[episode_index]
        src = SOURCE / f"data/chunk-{meta['data/chunk_index']:03d}/file-{meta['data/file_index']:03d}.parquet"
        table = pq.read_table(src, columns=["observation.state", "action"])
        state, action = vector(table, "observation.state"), vector(table, "action")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"non-finite source episode {episode_index}")
        groups[value].append(episode_index)

    rng = np.random.default_rng(SEED)
    selected = {value: sorted(int(x) for x in rng.choice(ids, PER_LABEL, replace=False)) for value, ids in groups.items()}
    assert len(selected[True]) == len(selected[False]) == PER_LABEL
    assert not set(selected[True]) & set(selected[False])
    return selected


def build(selected, info, labels, provenance, episodes, ep_schema):
    train = DEST / "train"
    for rel in ("data/chunk-000", "meta/episodes/chunk-000", "meta/annotations", "meta/alignments"):
        (train / rel).mkdir(parents=True, exist_ok=True)

    ordered = sorted(selected[True] + selected[False])
    meta_rows, label_rows, provenance_rows, mapping, checksums = [], [], [], [], []
    state_values, action_values, timestamps = [], [], []
    offset = 0
    video_features = [key for key, value in info["features"].items() if value.get("dtype") == "video"]

    for new_episode, old_episode in enumerate(ordered):
        old_meta = copy.deepcopy(episodes[old_episode])
        src_data = SOURCE / f"data/chunk-{old_meta['data/chunk_index']:03d}/file-{old_meta['data/file_index']:03d}.parquet"
        table = pq.read_table(src_data)
        n = len(table)
        assert n == old_meta["length"]
        state, action = vector(table, "observation.state"), vector(table, "action")
        state_values.append(state)
        action_values.append(action)
        timestamps.append(np.asarray(table["timestamp"])[:, None])
        table = table.set_column(table.schema.get_field_index("episode_index"), "episode_index", pa.array(np.full(n, new_episode, dtype=np.int64)))
        table = table.set_column(table.schema.get_field_index("index"), "index", pa.array(np.arange(offset, offset + n, dtype=np.int64)))
        dst_data = train / f"data/chunk-000/file-{new_episode:03d}.parquet"
        pq.write_table(table, dst_data, compression="zstd")
        checksums.append({"source": str(src_data.relative_to(SOURCE)), "source_sha256": digest(src_data), "target": str(dst_data.relative_to(train)), "target_sha256": digest(dst_data)})

        new_meta = copy.deepcopy(old_meta)
        new_meta.update(episode_index=new_episode, **{"data/chunk_index": 0, "data/file_index": new_episode}, dataset_from_index=offset, dataset_to_index=offset + n)
        for key in video_features:
            prefix = f"videos/{key}"
            src_video = SOURCE / prefix / f"chunk-{old_meta[prefix + '/chunk_index']:03d}" / f"file-{old_meta[prefix + '/file_index']:03d}.mp4"
            dst_video = train / prefix / "chunk-000" / f"file-{new_episode:03d}.mp4"
            hardlink(src_video, dst_video)
            new_meta[prefix + "/chunk_index"] = 0
            new_meta[prefix + "/file_index"] = new_episode
            checksums.append({"source": str(src_video.relative_to(SOURCE)), "source_sha256": digest(src_video), "target": str(dst_video.relative_to(train)), "target_sha256": digest(dst_video)})
        meta_rows.append(new_meta)

        label = copy.deepcopy(labels[old_episode])
        label.update(episode_index=new_episode, source_episode_index=old_episode, source_dataset_root=str(SOURCE), alignment_path=f"meta/alignments/episode-{new_episode:06d}.json")
        label_rows.append(label)
        prov = copy.deepcopy(provenance[old_episode])
        prov.update(episode_index=new_episode, source_episode_index=old_episode, source_dataset_root=str(SOURCE), alignment_path=f"meta/alignments/episode-{new_episode:06d}.json")
        provenance_rows.append(prov)
        src_align = SOURCE / provenance[old_episode]["alignment_path"]
        dst_align = train / f"meta/alignments/episode-{new_episode:06d}.json"
        hardlink(src_align, dst_align)
        mapping.append({"episode_index": new_episode, "source_episode_index": old_episode, "source_episode_id": provenance[old_episode]["source_episode_id"], "grasp_label_up": bool(label["extra_labels"]["self_annotation"]["grasp_label_up"]), "had_regrasp": int(label["extra_labels"]["self_annotation"]["had_regrasp"]), "frames": n})
        offset += n

    pq.write_table(pa.Table.from_pylist(meta_rows, schema=ep_schema), train / "meta/episodes/chunk-000/file-000.parquet", compression="zstd")
    write_jsonl(train / "meta/annotations/episode_labels.jsonl", label_rows)
    write_jsonl(train / "meta/episodes.jsonl", provenance_rows)
    write_jsonl(train / "episode_mapping.jsonl", mapping)
    write_jsonl(train / "file_checksums.jsonl", checksums)
    (train / "meta/tasks.parquet").write_bytes((SOURCE / "meta/tasks.parquet").read_bytes())

    all_state, all_action, all_time = np.concatenate(state_values), np.concatenate(action_values), np.concatenate(timestamps)
    split_info = copy.deepcopy(info)
    split_info.update(total_episodes=len(ordered), total_frames=int(offset), splits={"train": f"0:{len(ordered)}"}, source_dataset_root=str(SOURCE))
    write_json(train / "meta/info.json", split_info)
    write_json(train / "meta/stats.json", {"observation.state": stats(all_state), "action": stats(all_action), "timestamp": stats(all_time)})
    return train, mapping, all_state, all_action


def compute_norm(train: Path, mapping, states: np.ndarray, actions: np.ndarray) -> None:
    norm = {
        "observation.state.arm.position": stats(states[:, :14]),
        "observation.state.effector.position": stats(states[:, 14:]),
        "action.effector.position": stats(actions[:, 14:]),
    }
    arm = {}
    for horizon in range(HORIZON):
        values = []
        for row in mapping:
            table = pq.read_table(train / "data/chunk-000" / f"file-{int(row['episode_index']):03d}.parquet")
            state, action = vector(table, "observation.state"), vector(table, "action")
            idx = np.minimum(np.arange(len(action)) + horizon, len(action) - 1)
            values.append(action[idx, :14] - state[:, :14])
        for key, value in stats(np.concatenate(values)).items():
            arm.setdefault(key, []).append(value)
    norm["action.arm.position"] = arm
    out = DEST / "train_stats/norm_stats.json"
    write_json(out, {"norm_stats": norm, "count": int(len(states))})
    write_json(DEST / "train_stats/provenance.json", {"source_split": str(train), "train_episodes": len(mapping), "train_frames": int(len(states)), "test_episodes_used": 0, "chunk_size": HORIZON, "method": "balanced 75/75 selection; train-only exact NumPy statistics; arm action is future target minus current state with episode-tail hold", "sha256": digest(out)})


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite {DEST}")
    info, labels, provenance, episodes, ep_schema = load_source()
    selected = select_episodes(labels, episodes)
    train, mapping, states, actions = build(selected, info, labels, provenance, episodes, ep_schema)
    compute_norm(train, mapping, states, actions)
    label_counts = Counter("up" if row["grasp_label_up"] else "down" for row in mapping)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "destination": str(DEST),
        "dataset_name": DEST.name,
        "seed": SEED,
        "selection": "uniform reproducible sampling without replacement from each source label; complete episodes",
        "split_unit": "episode",
        "train": {"episodes": len(mapping), "frames": int(sum(row["frames"] for row in mapping)), "label_up": label_counts["up"], "label_down": label_counts["down"], "regrasp": int(sum(row["had_regrasp"] for row in mapping)), "source_episode_indices": [row["source_episode_index"] for row in mapping]},
        "videos_hardlinked": True,
        "source_labels_sha256": digest(SOURCE / "meta/annotations/episode_labels.jsonl"),
    }
    write_json(DEST / "manifest.json", manifest)
    write_json(DEST / "selection_report.json", {"seed": SEED, "per_label": PER_LABEL, "selected_source_episode_indices": {"label_up": selected[True], "label_down": selected[False]}, "source_total_by_label": {"label_up": sum(bool(x["extra_labels"]["self_annotation"]["grasp_label_up"]) for x in labels.values()), "label_down": sum(not bool(x["extra_labels"]["self_annotation"]["grasp_label_up"]) for x in labels.values())}})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
