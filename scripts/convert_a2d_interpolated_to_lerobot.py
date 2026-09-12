#!/usr/bin/env python3
"""Convert complete A2D raw episodes onto the left-wrist image timeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/a2d_interpolation_hf_cache")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/a2d_interpolation_hf_cache/datasets")

import cv2
import h5py
import numpy as np
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH


TASK = "Pick a soft package from the box, orient its label upward, and place it on the conveyor belt."
FPS = 30
VIDEO_KEYS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


def encode_h264_episode(video_key: str, episode_index: int, root: Path, fps: int) -> Path:
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    image_path = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    image_dir = (root / image_path).parent
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
              "-start_number", "0", "-i", str(image_dir / "frame-%06d.png"), "-pix_fmt", "yuv420p",
              "-movflags", "+faststart"]
    encoder = "h264_nvenc" if os.path.exists("/dev/nvidia0") else "libx264"
    result = subprocess.run([*common, "-c:v", encoder, *([] if encoder == "h264_nvenc" else ["-preset", "veryfast", "-crf", "18"]), str(temp_path)])
    if result.returncode and encoder == "h264_nvenc":
        subprocess.run([*common, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(temp_path)], check=True)
    elif result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    shutil.rmtree(image_dir)
    return temp_path


def jpeg_timestamps(directory: Path) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(directory.glob("*.jpg"), key=lambda p: int(p.stem))
    return np.asarray([int(p.stem) for p in paths], dtype=np.int64), paths


def manifest_timestamps(path: Path) -> np.ndarray:
    return np.asarray([int(line.split()[0]) for line in path.read_text().splitlines() if line], dtype=np.int64)


def nearest_indices(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    right = np.searchsorted(candidates, reference).clip(1, len(candidates) - 1)
    left = right - 1
    return np.where(np.abs(candidates[right] - reference) < np.abs(reference - candidates[left]), right, left)


def zoh_indices(reference: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    return np.searchsorted(sample_times, reference, side="right").clip(1, len(sample_times)) - 1


def interpolate(reference: np.ndarray, sample_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty((len(reference), values.shape[1]), dtype=np.float64)
    for dim in range(values.shape[1]):
        result[:, dim] = np.interp(reference, sample_times, values[:, dim])
    return result


def decode_head(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def read_jpeg(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unable to decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def features() -> dict:
    arm = [*[f"left_arm_joint_{i}_rad" for i in range(1, 8)], *[f"right_arm_joint_{i}_rad" for i in range(1, 8)]]
    return {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [*arm, "left_gripper_position", "right_gripper_position"]},
        "action": {"dtype": "float32", "shape": (16,), "names": [*[x.replace("_rad", "_target") for x in arm], "left_gripper_target", "right_gripper_target"]},
        "observation.images.cam_high_rgb": {"dtype": "video", "shape": (800, 1280, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam_left_wrist_rgb": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam_right_wrist_rgb": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
    }


def convert_episode(dataset: LeRobotDataset, episode: Path) -> dict:
    left_ts, left_paths = jpeg_timestamps(episode / "camera/hand_left/color")
    right_ts, right_paths = jpeg_timestamps(episode / "camera/hand_right/color")
    head_ts = manifest_timestamps(episode / "camera/head_color/head_color.txt")
    if not len(left_ts) or not len(right_ts) or not len(head_ts):
        raise RuntimeError(f"empty camera stream in {episode.name}")
    head_frames = decode_head(episode / "camera/head_color/head_color.h265")
    if len(head_frames) < len(head_ts):
        raise RuntimeError(f"head video has {len(head_frames)} frames but manifest has {len(head_ts)} in {episode.name}")
    with h5py.File(episode / "record/raw_joints.h5", "r") as h5:
        state_t = h5["state/joint/timestamp"][:]
        action_t = h5["action/joint/timestamp"][:]
        state_joint = h5["state/joint/position"][:]
        action_joint = h5["action/joint/position"][:]
        state_lt = h5["state/left_effector/timestamp"][:]
        state_rt = h5["state/right_effector/timestamp"][:]
        action_lt = h5["action/left_effector/timestamp"][:]
        action_rt = h5["action/right_effector/timestamp"][:]
        state_left = h5["state/left_effector/position"][:, 0]
        state_right = h5["state/right_effector/position"][:, 0]
        action_left = h5["action/left_effector/position"][:, 0]
        action_right = h5["action/right_effector/position"][:, 0]
        state_i = interpolate(left_ts, state_t, state_joint)
        action_i = interpolate(left_ts, action_t, action_joint)
        state_l_i = interpolate(left_ts, state_lt, state_left[:, None])[:, 0]
        state_r_i = interpolate(left_ts, state_rt, state_right[:, None])[:, 0]
        action_l_i = zoh_indices(left_ts, action_lt)
        action_r_i = zoh_indices(left_ts, action_rt)
        right_i = nearest_indices(left_ts, right_ts)
        head_i = nearest_indices(left_ts, head_ts)
        for i, left_path in enumerate(left_paths):
            state = np.concatenate([state_i[i], [state_l_i[i], state_r_i[i]]]).astype(np.float32)
            action = np.concatenate([action_i[i], [action_left[action_l_i[i]], action_right[action_r_i[i]]]]).astype(np.float32)
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise RuntimeError(f"non-finite interpolated value in {episode.name} frame {i}")
            dataset.add_frame({
                "observation.state": state,
                "action": action,
                "observation.images.cam_high_rgb": head_frames[int(head_i[i])],
                "observation.images.cam_left_wrist_rgb": read_jpeg(left_path),
                "observation.images.cam_right_wrist_rgb": read_jpeg(right_paths[int(right_i[i])]),
                "task": TASK,
            })
    dataset.save_episode(parallel_encoding=True)
    return {"source_id": episode.name, "frames": int(len(left_ts)), "source_start_ns": int(left_ts[0]), "source_end_ns": int(left_ts[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0, help="convert only this many episodes; 0 means all")
    parser.add_argument("--checkpoint-episodes", type=int, default=1, help="finalize and reopen after this many complete episodes")
    parser.add_argument("--resume", action="store_true", help="continue from a complete existing dataset prefix")
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    episodes = sorted(p for p in root.iterdir() if p.is_dir())
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    if args.checkpoint_episodes < 1:
        raise ValueError("--checkpoint-episodes must be positive")
    lerobot_dataset_module._encode_video_worker = encode_h264_episode
    if args.resume:
        dataset = LeRobotDataset(repo_id=output.name, root=output, video_backend="pyav")
        start_episode = dataset.meta.total_episodes
    else:
        dataset = LeRobotDataset.create(repo_id=output.name, root=output, fps=FPS, features=features(), robot_type="a2d_bimanual_arm", use_videos=True, image_writer_threads=16, video_backend="pyav")
        start_episode = 0
    if start_episode > len(episodes):
        raise RuntimeError(f"output has {start_episode} episodes but source has only {len(episodes)}")
    converted = []
    try:
        for n, episode in enumerate(episodes[start_episode:], start_episode + 1):
            report = convert_episode(dataset, episode)
            converted.append(report)
            if len(converted) % args.checkpoint_episodes == 0:
                dataset.finalize()
                dataset = LeRobotDataset(repo_id=output.name, root=output, video_backend="pyav")
            if n % 10 == 0 or n == len(episodes):
                print(json.dumps({"progress": n, "total": len(episodes), "frames_since_resume": sum(x["frames"] for x in converted)}), flush=True)
    finally:
        dataset.finalize()
    report = {"source_root": str(root), "timeline": "left_wrist_jpeg_timestamp", "state_alignment": "linear_interpolation", "action_joint_alignment": "linear_interpolation", "gripper_alignment": "zero_order_hold", "video_alignment": "nearest_frame_without_drop", "source_episodes": len(episodes), "output_episodes": start_episode + len(converted), "output_frames": sum(x["frames"] for x in converted), "episodes_converted_this_run": converted}
    (output / "interpolation_conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("source_episodes", "output_episodes", "output_frames")}, indent=2))


if __name__ == "__main__":
    main()
