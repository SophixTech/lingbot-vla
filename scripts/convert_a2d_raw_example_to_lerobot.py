#!/usr/bin/env python3
"""Convert timestamped raw A2D data into a conservative LeRobot v3 episode.

This converter is intentionally limited to the signals present in an A2D raw
export: head RGB, two wrist RGB cameras, 14 arm joints, and two grippers.  It
does not invent labels for missing data.  A left-wrist frame is retained only
when every selected stream has a nearest timestamp within ``--tolerance-ms``.
The output therefore remains suitable for validating the VLA data path even
when an official A2D alignment product is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK = "Pick a soft package from the box, orient its label upward, and place it on the conveyor belt."
FPS = 30


def timestamps_from_jpegs(directory: Path) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))
    return np.asarray([int(path.stem) for path in paths], dtype=np.int64), paths


def timestamps_from_manifest(path: Path) -> np.ndarray:
    return np.asarray([int(line.split()[0]) for line in path.read_text().splitlines() if line], dtype=np.int64)


def nearest_indices(reference: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(candidates, reference).clip(1, len(candidates) - 1)
    left = right - 1
    choose_right = np.abs(candidates[right] - reference) < np.abs(reference - candidates[left])
    indices = np.where(choose_right, right, left)
    return indices, np.abs(candidates[indices] - reference)


def decode_video_frames(video_path: Path, frame_indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    result: dict[int, np.ndarray] = {}
    index = 0
    while index <= max(frame_indices):
        ok, frame = capture.read()
        if not ok:
            break
        if index in frame_indices:
            result[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    missing = frame_indices - result.keys()
    if missing:
        raise RuntimeError(f"Unable to decode head RGB frames: {sorted(missing)[:10]}")
    return result


def read_jpeg(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=17.0)
    args = parser.parse_args()

    raw = args.raw.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    tolerance_ns = int(args.tolerance_ms * 1_000_000)

    left_ts, left_paths = timestamps_from_jpegs(raw / "camera/hand_left/color")
    right_ts, right_paths = timestamps_from_jpegs(raw / "camera/hand_right/color")
    head_ts = timestamps_from_manifest(raw / "camera/head_color/head_color.txt")

    with h5py.File(raw / "record/raw_joints.h5", "r") as h5:
        state_joint_ts = h5["state/joint/timestamp"][:]
        action_joint_ts = h5["action/joint/timestamp"][:]
        action_left_gripper_ts = h5["action/left_effector/timestamp"][:]
        action_right_gripper_ts = h5["action/right_effector/timestamp"][:]

        mappings = {
            "right_wrist": nearest_indices(left_ts, right_ts),
            "head_rgb": nearest_indices(left_ts, head_ts),
            "state_joint": nearest_indices(left_ts, state_joint_ts),
            "action_joint": nearest_indices(left_ts, action_joint_ts),
            "action_left_gripper": nearest_indices(left_ts, action_left_gripper_ts),
            "action_right_gripper": nearest_indices(left_ts, action_right_gripper_ts),
        }
        valid = np.ones(len(left_ts), dtype=bool)
        for _, (_, errors) in mappings.items():
            valid &= errors <= tolerance_ns
        selected = np.flatnonzero(valid)
        if len(selected) == 0:
            raise RuntimeError("No frames satisfy the requested timestamp tolerance.")

        head_indices = set(mappings["head_rgb"][0][selected].tolist())
        head_frames = decode_video_frames(raw / "camera/head_color/head_color.h265", head_indices)

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": [
                    *[f"left_arm_joint_{i}_rad" for i in range(1, 8)],
                    *[f"right_arm_joint_{i}_rad" for i in range(1, 8)],
                    "left_gripper_position",
                    "right_gripper_position",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (16,),
                "names": [
                    *[f"left_arm_joint_{i}_target" for i in range(1, 8)],
                    *[f"right_arm_joint_{i}_target" for i in range(1, 8)],
                    "left_gripper_target",
                    "right_gripper_target",
                ],
            },
            "observation.images.cam_high_rgb": {
                "dtype": "video",
                "shape": (800, 1280, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_left_wrist_rgb": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_right_wrist_rgb": {
                "dtype": "video",
                "shape": (480, 848, 3),
                "names": ["height", "width", "channels"],
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=output.name,
            root=output,
            fps=FPS,
            features=features,
            robot_type="a2d_bimanual_arm",
            use_videos=True,
            image_writer_threads=4,
            video_backend="pyav",
        )

        for frame_index in selected:
            state_i = mappings["state_joint"][0][frame_index]
            action_i = mappings["action_joint"][0][frame_index]
            left_gripper_i = mappings["action_left_gripper"][0][frame_index]
            right_gripper_i = mappings["action_right_gripper"][0][frame_index]
            right_i = mappings["right_wrist"][0][frame_index]
            head_i = mappings["head_rgb"][0][frame_index]

            state = np.concatenate(
                [
                    h5["state/joint/position"][state_i],
                    h5["state/left_effector/position"][state_i],
                    h5["state/right_effector/position"][state_i],
                ]
            ).astype(np.float32)
            action = np.concatenate(
                [
                    h5["action/joint/position"][action_i],
                    h5["action/left_effector/position"][left_gripper_i],
                    h5["action/right_effector/position"][right_gripper_i],
                ]
            ).astype(np.float32)
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "observation.images.cam_high_rgb": head_frames[head_i],
                    "observation.images.cam_left_wrist_rgb": read_jpeg(left_paths[frame_index]),
                    "observation.images.cam_right_wrist_rgb": read_jpeg(right_paths[right_i]),
                    "task": TASK,
                }
            )
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()

    report = {
        "source": str(raw),
        "selection_policy": {
            "anchor": "camera/hand_left/color JPEG timestamps",
            "required_streams": list(mappings),
            "tolerance_ms": args.tolerance_ms,
            "state_action_dimensions": 16,
            "excluded_controls": ["head", "waist", "base"],
        },
        "source_frame_counts": {
            "left_wrist": len(left_ts),
            "right_wrist": len(right_ts),
            "head_rgb": len(head_ts),
        },
        "output_frames": int(len(selected)),
        "source_time_range_ns": [int(left_ts[selected[0]]), int(left_ts[selected[-1]])],
        "max_match_error_ms": {
            key: float(errors[selected].max() / 1_000_000) for key, (_, errors) in mappings.items()
        },
        "median_match_error_ms": {
            key: float(np.median(errors[selected]) / 1_000_000) for key, (_, errors) in mappings.items()
        },
        "task": TASK,
        "known_limitations": [
            "Timestamp-nearest matching is an inferred alignment, not an official aligned_joints product.",
            "Joint action units and command semantics require robot-side confirmation.",
        ],
    }
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
