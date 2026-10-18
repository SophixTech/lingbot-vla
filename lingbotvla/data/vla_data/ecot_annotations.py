"""Reader for sidecar ECOT annotations."""

import json
from pathlib import Path


SUBTASK_TO_ID = {
    "pick_soft_package": 0,
    "place_label_up": 1,
    "flip_and_place_label_up": 2,
    "retract_or_idle": 3,
}


class ECOTAnnotations:
    """Load episode annotations without copying or modifying source data."""

    def __init__(self, root: str | Path, split: str, source_dataset: str | Path):
        self.root = Path(root)
        self.split = split
        self.records: dict[tuple[str, int], dict] = {}
        self.image_sizes: dict[tuple[str, int], tuple[int, int]] = {}
        source_meta = {}
        meta_path = Path(source_dataset) / split / "meta/episodes.jsonl"
        if meta_path.exists():
            source_meta = {int(row["episode_index"]): row for row in
                           (json.loads(line) for line in meta_path.read_text().splitlines() if line.strip())}
        index_path = self.root / "annotations.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"ECOT annotation index not found: {index_path}")

        with index_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                index = json.loads(line)
                if "source_dataset" in index and Path(index["source_dataset"]).resolve() != Path(source_dataset).resolve():
                    raise ValueError(f"{index_path}:{line_no}: source dataset mismatch")
                if "source_dataset" not in index:
                    meta = source_meta.get(int(index["episode_index"])) if index["split"] == split else None
                    if (meta is None or not index.get("source_episode_id") or
                            index["source_episode_id"] != meta.get("source_episode_id") or
                            int(index["frame_count"]) != int(meta["frame_count"])):
                        raise ValueError(f"{index_path}:{line_no}: source episode identity/frame mismatch")
                key = (str(index["split"]), int(index["episode_index"]))
                if key in self.records:
                    raise ValueError(f"{index_path}:{line_no}: duplicate {key}")
                annotation_path = self.root / index["annotation_path"]
                record = json.loads(annotation_path.read_text(encoding="utf-8"))
                self._validate(index, record, annotation_path)
                self.records[key] = record
                self.image_sizes[key] = tuple(index["image_size"])

    @property
    def episode_indices(self) -> list[int]:
        return sorted(ep for split, ep in self.records if split == self.split)

    def get(self, episode_index: int) -> dict | None:
        return self.records.get((self.split, int(episode_index)))

    def target(self, episode_index: int, frame_index: int) -> tuple[int, list[float], bool, int]:
        """Return subtask id, normalized bbox, bbox-valid flag, and label id."""
        record = self.get(episode_index)
        if record is None:
            return -100, [0.0, 0.0, 0.0, 0.0], False, -100

        frame_index = int(frame_index)
        segments = record["subtask"]
        subtask_name = next(
            segment["subtask"]
            for segment in segments
            if segment["start_frame"] <= frame_index < segment["end_frame_exclusive"]
        )
        bbox = record["bbox"][frame_index]
        if bbox is None:
            bbox_target = [0.0, 0.0, 0.0, 0.0]
            bbox_valid = False
        else:
            width, height = self.image_sizes[(self.split, int(episode_index))]
            bbox_target = [
                float(bbox[0]) / width,
                float(bbox[1]) / height,
                float(bbox[2]) / width,
                float(bbox[3]) / height,
            ]
            bbox_valid = True
        label_target = 1 if bool(record["grasp_label_up"]) else 0
        return SUBTASK_TO_ID[subtask_name], bbox_target, bbox_valid, label_target

    @staticmethod
    def _validate(index: dict, record: dict, path: Path) -> None:
        required = {"subtask", "bbox", "grasp_label_up"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        frame_count = int(index["frame_count"])
        width, height = index["image_size"]
        if len(record["bbox"]) != frame_count:
            raise ValueError(f"{path}: bbox length {len(record['bbox'])} != {frame_count}")
        segments = record["subtask"]
        if not segments or segments[0]["start_frame"] != 0 or segments[-1]["end_frame_exclusive"] != frame_count:
            raise ValueError(f"{path}: subtask does not cover [0, {frame_count})")
        for previous, current in zip(segments, segments[1:]):
            if previous["end_frame_exclusive"] != current["start_frame"]:
                raise ValueError(f"{path}: subtask gap or overlap")
        for bbox in record["bbox"]:
            if bbox is None:
                continue
            if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= width and 0 <= bbox[1] < bbox[3] <= height):
                raise ValueError(f"{path}: bbox outside image bounds")
