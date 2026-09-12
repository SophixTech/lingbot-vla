#!/usr/bin/env python3
"""Prepare the three-camera AgiBot parts dataset for LingBot-VLA.

The RoboCOIN source is LeRobot v2.1 and advertises eight cameras.  The
single-GPU baseline intentionally downloads only the top and two wrist cameras.
This tool creates a hard-link staging copy, adjusts *only its* metadata to the
three available cameras, and calls LeRobot's official v2.1-to-v3 converter.
The original download is never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset


CAMERAS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def load_info(source: Path) -> dict:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}. Download meta/* first.")
    return json.loads(info_path.read_text())


def validate_source(source: Path, info: dict) -> None:
    expected_episodes = int(info["total_episodes"])
    parquet_files = list((source / "data").glob("chunk-*/*.parquet"))
    if len(parquet_files) != expected_episodes:
        raise RuntimeError(
            f"Expected {expected_episodes} parquet episodes but found {len(parquet_files)} in {source / 'data'}."
        )

    for camera in CAMERAS:
        videos = list((source / "videos").glob(f"chunk-*/{camera}/episode_*.mp4"))
        if len(videos) != expected_episodes:
            raise RuntimeError(
                f"Expected {expected_episodes} videos for {camera} but found {len(videos)}. "
                "Resume the dataset download before converting."
            )

    required_meta = ("episodes.jsonl", "episodes_stats.jsonl", "info.json", "tasks.jsonl")
    missing = [name for name in required_meta if not (source / "meta" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required v2.1 metadata: {', '.join(missing)}")


def create_staging_copy(source: Path, target: Path, info: dict) -> None:
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite {target}. Remove or rename it only after inspecting its contents."
        )
    target.mkdir(parents=True)
    # Metadata and annotations are small and must remain independently
    # writable. Do not copy the raw data tree before hard-linking it: that
    # would briefly consume an extra 18 GiB.
    for relative in ("README.md", "meta", "annotations"):
        source_path = source / relative
        if source_path.is_dir():
            shutil.copytree(source_path, target / relative, copy_function=shutil.copy2)
        elif source_path.is_file():
            shutil.copy2(source_path, target / relative)

    # The converted v3 files are new files, but v2.1 data and source videos
    # are immutable input and can share disk blocks with the raw download.
    for relative in ("data", "videos"):
        shutil.copytree(source / relative, target / relative, copy_function=os.link)

    staged_info_path = target / "meta" / "info.json"
    staged_info = json.loads(staged_info_path.read_text())
    staged_info["features"] = {
        key: value
        for key, value in staged_info["features"].items()
        if value.get("dtype") != "video" or key in CAMERAS
    }
    staged_info_path.write_text(json.dumps(staged_info, indent=2) + "\n")


def align_feature_order_to_parquet(target: Path) -> None:
    """Make the v3 metadata's non-video feature order match its Arrow files.

    LeRobot's v3 loader casts Arrow tables against the feature mapping and
    requires field order to match exactly. The upstream v2.1-to-v3 converter
    can preserve source JSON order instead of the rewritten parquet order.
    """
    info_path = target / "meta" / "info.json"
    data_files = sorted((target / "data").glob("chunk-*/*.parquet"))
    if not info_path.is_file() or not data_files:
        raise FileNotFoundError("Expected converted meta/info.json and data parquet files.")

    info = json.loads(info_path.read_text())
    features = info["features"]
    parquet_names = pq.ParquetFile(data_files[0]).schema_arrow.names
    for data_file in data_files[1:]:
        if pq.ParquetFile(data_file).schema_arrow.names != parquet_names:
            raise RuntimeError(f"Inconsistent parquet schema in {data_file}")

    video_names = [key for key, value in features.items() if value.get("dtype") == "video"]
    non_video_names = [key for key in features if key not in video_names]
    # This source's parquet contains a gripper activity action annotation that
    # its v2.1 metadata omits. The converter preserves the column and stats,
    # so complete the missing v3 feature definition rather than dropping data.
    unexpected_columns = [key for key in parquet_names if key not in non_video_names]
    if unexpected_columns:
        if unexpected_columns != ["gripper_activity_action"] or "gripper_activity_state" not in features:
            raise RuntimeError(f"Parquet columns missing from metadata: {unexpected_columns}")
        action_feature = copy.deepcopy(features["gripper_activity_state"])
        action_feature["names"] = ["left_gripper_activity", "right_gripper_activity"]
        features["gripper_activity_action"] = action_feature
        non_video_names.append("gripper_activity_action")
    if set(non_video_names) != set(parquet_names):
        raise RuntimeError("Metadata non-video feature names do not match parquet columns.")

    # Keep video metadata first for human readability; the loader excludes it
    # when constructing the Arrow schema. All retained table columns follow
    # the exact parquet order.
    info["features"] = {key: features[key] for key in [*video_names, *parquet_names]}
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="Downloaded RoboCOIN v2.1 directory")
    parser.add_argument("--output", type=Path, required=True, help="Output LeRobot v3 directory")
    parser.add_argument("--check-only", action="store_true", help="Validate download completeness without writing")
    parser.add_argument("--repair-existing", action="store_true", help="Repair feature order in an existing v3 output")
    args = parser.parse_args()

    source = args.raw.resolve()
    target = args.output.resolve()
    if args.repair_existing:
        align_feature_order_to_parquet(target)
        print(f"Aligned v3 metadata schema in {target}")
        return
    info = load_info(source)
    validate_source(source, info)
    print(f"Validated {info['total_episodes']} episodes and {len(CAMERAS)} cameras in {source}")
    if args.check_only:
        return

    create_staging_copy(source, target, info)
    convert_dataset(
        repo_id=target.name,
        root=target.parent,
        data_file_size_in_mb=100,
        video_file_size_in_mb=500,
        push_to_hub=False,
    )
    align_feature_order_to_parquet(target)
    print(f"Converted dataset written to {target}")


if __name__ == "__main__":
    main()
