#!/usr/bin/env python3
"""Repair v3 feature metadata order without importing the converter stack."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    info_path = root / "meta" / "info.json"
    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not info_path.is_file() or not data_files:
        raise FileNotFoundError("Expected v3 meta/info.json and data parquet files")

    info = json.loads(info_path.read_text())
    features = info["features"]
    parquet_names = pq.ParquetFile(data_files[0]).schema_arrow.names
    if any(pq.ParquetFile(path).schema_arrow.names != parquet_names for path in data_files[1:]):
        raise RuntimeError("Converted parquet files do not share a schema")

    video_names = [name for name, feature in features.items() if feature.get("dtype") == "video"]
    non_video_names = [name for name in features if name not in video_names]
    missing = [name for name in parquet_names if name not in non_video_names]
    if missing:
        if missing != ["gripper_activity_action"] or "gripper_activity_state" not in features:
            raise RuntimeError(f"Parquet columns missing from metadata: {missing}")
        features["gripper_activity_action"] = copy.deepcopy(features["gripper_activity_state"])
        features["gripper_activity_action"]["names"] = ["left_gripper_activity", "right_gripper_activity"]
        non_video_names.append("gripper_activity_action")

    extra = [name for name in non_video_names if name not in parquet_names]
    if extra:
        raise RuntimeError(f"Metadata columns missing from parquet: {extra}")

    info["features"] = {name: features[name] for name in [*video_names, *parquet_names]}
    info_path.write_text(json.dumps(info, indent=2) + "\n")

    # The source stores this label as an int32[1] list, while the v3 feature
    # schema defines it as a scalar. Convert only this derived, auxiliary
    # label column; all sample values and every other column are retained.
    for data_file in data_files:
        table = pq.read_table(data_file)
        field_index = table.schema.get_field_index("scene_annotation")
        field_type = table.schema.field("scene_annotation").type
        if not str(field_type).startswith("list<"):
            continue
        scalar = pc.list_element(table["scene_annotation"], 0).cast(field_type.value_type)
        repaired = table.set_column(field_index, "scene_annotation", scalar)
        temporary = data_file.with_suffix(data_file.suffix + ".tmp")
        pq.write_table(repaired, temporary, compression="zstd")
        os.replace(temporary, data_file)

    print(f"Aligned {len(parquet_names)} non-video features in {info_path}")


if __name__ == "__main__":
    main()
