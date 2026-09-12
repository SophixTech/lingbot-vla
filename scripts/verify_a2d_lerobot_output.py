#!/usr/bin/env python3
"""Verify a completed A2D->LeRobot v3 conversion without changing source data."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROOT = Path("/home/bjtc/Sophix/datasets/a2d_bag_record_v3")
EXPECTED_EPISODES = 2448
EXPECTED_FRAMES = 390335
VIDEO_KEYS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    report = json.loads((ROOT / "conversion_report.json").read_text())
    info = json.loads((ROOT / "meta/info.json").read_text())
    if (info["total_episodes"], info["total_frames"]) != (EXPECTED_EPISODES, EXPECTED_FRAMES):
        fail(f"info count mismatch: {(info['total_episodes'], info['total_frames'])}")
    if (report["output_episodes"], report["output_frames"]) != (EXPECTED_EPISODES, EXPECTED_FRAMES):
        fail(f"plan count mismatch: {(report['output_episodes'], report['output_frames'])}")

    episode_rows = []
    for path in sorted((ROOT / "meta/episodes").glob("*/*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    if len(episode_rows) != EXPECTED_EPISODES:
        fail(f"episode metadata rows={len(episode_rows)}")
    expected_from = 0
    for expected_episode, row in enumerate(episode_rows):
        if row["episode_index"] != expected_episode or row["dataset_from_index"] != expected_from:
            fail(f"episode continuity error at {expected_episode}")
        expected_from = row["dataset_to_index"]
    if expected_from != EXPECTED_FRAMES:
        fail(f"episode metadata final frame={expected_from}")

    data_rows = 0
    for path in sorted((ROOT / "data").glob("*/*.parquet")):
        data_rows += pq.ParquetFile(path).metadata.num_rows
    if data_rows != EXPECTED_FRAMES:
        fail(f"data parquet rows={data_rows}")

    for key in VIDEO_KEYS:
        paths = sorted((ROOT / "videos" / key).glob("*/*.mp4"))
        if not paths:
            fail(f"no videos for {key}")
        for path in paths:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1", str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode or not result.stdout.startswith("duration="):
                fail(f"undecodable video {path}: {result.stderr.strip()}")

    dataset = LeRobotDataset(repo_id=ROOT.name, root=ROOT, video_backend="pyav")
    if (dataset.num_episodes, dataset.num_frames) != (EXPECTED_EPISODES, EXPECTED_FRAMES):
        fail(f"dataset reader count mismatch: {(dataset.num_episodes, dataset.num_frames)}")
    sample_indices = (0, dataset.num_frames // 2, dataset.num_frames - 1)
    for index in sample_indices:
        item = dataset[index]
        for key in VIDEO_KEYS:
            if tuple(item[key].shape) not in ((3, 800, 1280), (3, 480, 848)):
                fail(f"decoded shape {key}={tuple(item[key].shape)} at frame {index}")
        if tuple(item["observation.state"].shape) != (16,) or tuple(item["action"].shape) != (16,):
            fail(f"vector shape error at frame {index}")

    result = {
        "verified_episodes": EXPECTED_EPISODES,
        "verified_frames": EXPECTED_FRAMES,
        "video_files": {key: len(list((ROOT / "videos" / key).glob("*/*.mp4"))) for key in VIDEO_KEYS},
        "sample_frames_read": list(sample_indices),
    }
    (ROOT / "verification_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        raise
