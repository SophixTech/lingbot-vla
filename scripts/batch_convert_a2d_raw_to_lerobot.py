#!/usr/bin/env python3
"""Conservatively group raw A2D episodes and write a LeRobot v3 dataset.

The source export is not an official aligned product.  This tool uses the
left-wrist JPEG timestamp as its anchor and requires nearest right-wrist RGB,
head RGB, arm state, arm action, and both gripper actions to fall inside one
strict tolerance window.  It never interpolates missing actions.  Disjoint
valid spans become separate LeRobot episodes, avoiding temporal jumps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH
from lerobot.datasets.video_utils import encode_video_frames


TASK = "Pick a soft package from the box, orient its label upward, and place it on the conveyor belt."
FPS = 30


def encode_h264_episode(video_key: str, episode_index: int, root: Path, fps: int) -> Path:
    """Use H.264 instead of LeRobot's much slower default AV1 encoder."""
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    frame_path = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    image_dir = (root / frame_path).parent
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-start_number", "0",
            "-i", str(image_dir / "frame-%06d.png"),
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temp_path),
        ],
        check=True,
    )
    shutil.rmtree(image_dir)
    return temp_path


def jpeg_timestamps(directory: Path) -> tuple[np.ndarray, list[Path]]:
    paths = sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))
    return np.asarray([int(path.stem) for path in paths], dtype=np.int64), paths


def manifest_timestamps(path: Path) -> np.ndarray:
    return np.asarray([int(line.split()[0]) for line in path.read_text().splitlines() if line], dtype=np.int64)


def nearest(reference: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(candidates):
        raise ValueError("empty timestamp stream")
    right = np.searchsorted(candidates, reference).clip(1, len(candidates) - 1)
    left = right - 1
    use_right = np.abs(candidates[right] - reference) < np.abs(reference - candidates[left])
    indices = np.where(use_right, right, left)
    return indices, np.abs(candidates[indices] - reference)


def required_paths(episode: Path) -> list[Path]:
    return [
        episode / "camera/hand_left/color",
        episode / "camera/hand_right/color",
        episode / "camera/head_color/head_color.txt",
        episode / "camera/head_color/head_color.h265",
        episode / "record/raw_joints.h5",
    ]


def scan_episode(episode: Path, tolerance_ns: int, min_frames: int) -> dict:
    missing = [str(path.relative_to(episode)) for path in required_paths(episode) if not path.exists()]
    if missing:
        return {"id": episode.name, "status": "rejected", "reason": f"missing: {', '.join(missing)}"}

    left_ts, left_paths = jpeg_timestamps(episode / "camera/hand_left/color")
    right_ts, right_paths = jpeg_timestamps(episode / "camera/hand_right/color")
    head_ts = manifest_timestamps(episode / "camera/head_color/head_color.txt")
    if not len(left_ts) or not len(right_ts) or not len(head_ts):
        return {"id": episode.name, "status": "rejected", "reason": "empty camera stream"}

    try:
        with h5py.File(episode / "record/raw_joints.h5", "r") as h5:
            streams = {
                "state_joint": h5["state/joint/timestamp"][:],
                "action_joint": h5["action/joint/timestamp"][:],
                "action_left_gripper": h5["action/left_effector/timestamp"][:],
                "action_right_gripper": h5["action/right_effector/timestamp"][:],
            }
    except Exception as exc:
        return {"id": episode.name, "status": "rejected", "reason": f"HDF5 read error: {exc}"}
    if not all(len(values) for values in streams.values()):
        empty = [key for key, values in streams.items() if not len(values)]
        return {"id": episode.name, "status": "rejected", "reason": f"empty required stream: {', '.join(empty)}"}

    mappings = {
        "right_wrist": nearest(left_ts, right_ts),
        "head_rgb": nearest(left_ts, head_ts),
        **{name: nearest(left_ts, values) for name, values in streams.items()},
    }
    valid = np.ones(len(left_ts), dtype=bool)
    for _, errors in mappings.values():
        valid &= errors <= tolerance_ns
    selected = np.flatnonzero(valid)
    if not len(selected):
        return {"id": episode.name, "status": "rejected", "reason": "no frames within tolerance"}

    # A long anchor-timestamp gap indicates dropped video frames. Do not join
    # data on opposite sides of it even if their source indices are adjacent.
    source_gaps = np.diff(selected) > 1
    timestamp_gaps = np.diff(left_ts[selected]) > int(1.5 / FPS * 1_000_000_000)
    split_at = np.flatnonzero(source_gaps | timestamp_gaps) + 1
    runs = [run for run in np.split(selected, split_at) if len(run) >= min_frames]
    if not runs:
        return {"id": episode.name, "status": "rejected", "reason": f"no contiguous run >= {min_frames} frames"}

    return {
        "id": episode.name,
        "status": "ready",
        "left_paths": left_paths,
        "right_paths": right_paths,
        "mappings": mappings,
        "runs": runs,
        "source_frames": int(len(left_ts)),
        "selected_frames": int(sum(len(run) for run in runs)),
        "segments": [int(len(run)) for run in runs],
        "max_match_error_ms": {key: float(errors[np.concatenate(runs)].max() / 1_000_000) for key, (_, errors) in mappings.items()},
    }


def decode_jpeg(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"JPEG decode failed: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class HeadReader:
    """Sequential decoder that reuses a decoded head frame for duplicate matches."""

    def __init__(self, path: Path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open head RGB video: {path}")
        self.index = -1
        self.frame: np.ndarray | None = None

    def get(self, target: int) -> np.ndarray:
        while self.index < target:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(f"Head RGB ends before manifest frame {target}")
            self.index += 1
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.index != target or self.frame is None:
            raise RuntimeError(f"Head RGB cannot seek backwards from {self.index} to {target}")
        return self.frame

    def close(self) -> None:
        self.capture.release()


def features() -> dict:
    names = [*[f"left_arm_joint_{i}_rad" for i in range(1, 8)], *[f"right_arm_joint_{i}_rad" for i in range(1, 8)]]
    return {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [*names, "left_gripper_position", "right_gripper_position"]},
        "action": {"dtype": "float32", "shape": (16,), "names": [*[name.replace("_rad", "_target") for name in names], "left_gripper_target", "right_gripper_target"]},
        "observation.images.cam_high_rgb": {"dtype": "video", "shape": (800, 1280, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam_left_wrist_rgb": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
        "observation.images.cam_right_wrist_rgb": {"dtype": "video", "shape": (480, 848, 3), "names": ["height", "width", "channels"]},
    }


def recover_interrupted_output(output: Path) -> dict:
    """Roll an interrupted v3 output back to its last complete parquet commit."""
    episode_files = sorted((output / "meta/episodes").glob("*/*.parquet"))
    accepted_files: list[Path] = []
    accepted_rows: list[dict] = []
    expected_episode = 0
    expected_frame = 0
    data_rows: dict[Path, list[dict]] = {}

    for episode_file in episode_files:
        try:
            rows = pq.read_table(episode_file).to_pylist()
        except Exception:
            break
        candidate_episode = expected_episode
        candidate_frame = expected_frame
        candidate_data_rows = {path: list(values) for path, values in data_rows.items()}
        for row in rows:
            if row["episode_index"] != candidate_episode or row["dataset_from_index"] != candidate_frame:
                rows = []
                break
            data_path = output / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
            candidate_data_rows.setdefault(data_path, []).append(row)
            candidate_episode += 1
            candidate_frame = row["dataset_to_index"]
        if rows:
            for data_path, related_rows in candidate_data_rows.items():
                try:
                    first = min(row["dataset_from_index"] for row in related_rows)
                    last = max(row["dataset_to_index"] for row in related_rows)
                    if pq.ParquetFile(data_path).metadata.num_rows != last - first:
                        rows = []
                        break
                except Exception:
                    rows = []
                    break
        if not rows:
            break
        accepted_files.append(episode_file)
        accepted_rows.extend(rows)
        data_rows = candidate_data_rows
        expected_episode = candidate_episode
        expected_frame = candidate_frame

    if not accepted_rows:
        raise RuntimeError("No complete LeRobot episode metadata commit remains; refusing destructive recovery")

    keep_data = {
        output / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
        for row in accepted_rows
    }
    keep_video = set()
    for row in accepted_rows:
        for video_key in features():
            if features()[video_key]["dtype"] != "video":
                continue
            keep_video.add(
                output / "videos" / video_key / f"chunk-{row[f'videos/{video_key}/chunk_index']:03d}"
                / f"file-{row[f'videos/{video_key}/file_index']:03d}.mp4"
            )

    removed = {"data": 0, "episode_metadata": 0, "video": 0, "images": 0}
    archive_root = output / "interrupted_artifacts" / f"before_recovery_episode_{expected_episode:06d}"

    def archive(path: Path) -> None:
        destination = archive_root / path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, destination)

    for path in (output / "data").glob("*/*.parquet"):
        if path not in keep_data:
            archive(path)
            removed["data"] += 1
    for path in episode_files:
        if path not in accepted_files:
            archive(path)
            removed["episode_metadata"] += 1
    for path in (output / "videos").glob("*/*/*.mp4"):
        if path not in keep_video:
            archive(path)
            removed["video"] += 1
    for image_root in (output / "images").glob("*"):
        for image_dir in image_root.glob("episode-*"):
            try:
                episode_index = int(image_dir.name.removeprefix("episode-"))
            except ValueError:
                continue
            if episode_index >= expected_episode:
                archive(image_dir)
                removed["images"] += 1

    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = expected_episode
    info["total_frames"] = expected_frame
    info["splits"] = {"train": f"0:{expected_episode}"}
    info_path.write_text(json.dumps(info, indent=4) + "\n")

    # Incremental statistics include uncommitted frames after an interruption.
    # The final conversion verification recomputes them from the completed data.
    stats_path = output / "meta/stats.json"
    if stats_path.exists():
        archive(stats_path)
        removed["stats"] = 1
    return {"episodes": expected_episode, "frames": expected_frame, "removed": removed, "archive": str(archive_root) if any(removed.values()) else None}


def convert_episode(dataset: LeRobotDataset, episode: Path, plan: dict) -> list[dict]:
    mappings = plan["mappings"]
    reports = []
    with h5py.File(episode / "record/raw_joints.h5", "r") as h5:
        state_joint = h5["state/joint/position"][:]
        state_left = h5["state/left_effector/position"][:]
        state_right = h5["state/right_effector/position"][:]
        action_joint = h5["action/joint/position"][:]
        action_left = h5["action/left_effector/position"][:]
        action_right = h5["action/right_effector/position"][:]
        reader = HeadReader(episode / "camera/head_color/head_color.h265")
        try:
            for run in plan["runs"]:
                for source_index in run:
                    state_i = mappings["state_joint"][0][source_index]
                    action_i = mappings["action_joint"][0][source_index]
                    left_action_i = mappings["action_left_gripper"][0][source_index]
                    right_action_i = mappings["action_right_gripper"][0][source_index]
                    right_i = mappings["right_wrist"][0][source_index]
                    head_i = mappings["head_rgb"][0][source_index]
                    state = np.concatenate([state_joint[state_i], state_left[state_i], state_right[state_i]]).astype(np.float32)
                    action = np.concatenate([action_joint[action_i], action_left[left_action_i], action_right[right_action_i]]).astype(np.float32)
                    if not np.isfinite(state).all() or not np.isfinite(action).all():
                        raise RuntimeError(f"non-finite value at source frame {source_index}")
                    dataset.add_frame({
                        "observation.state": state,
                        "action": action,
                        "observation.images.cam_high_rgb": reader.get(int(head_i)),
                        "observation.images.cam_left_wrist_rgb": decode_jpeg(plan["left_paths"][source_index]),
                        "observation.images.cam_right_wrist_rgb": decode_jpeg(plan["right_paths"][right_i]),
                        "task": TASK,
                    })
                output_index = dataset.meta.total_episodes
                # The three camera streams are independent at encoding time.
                # Encoding them concurrently uses the available host CPU cores
                # without changing frame ordering or timestamp metadata.
                dataset.save_episode(parallel_encoding=True)
                reports.append({"source_id": episode.name, "output_episode": output_index, "frames": int(len(run))})
        finally:
            reader.close()
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "convert"), default="plan")
    parser.add_argument("--tolerance-ms", type=float, default=17.0)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--checkpoint-episodes", type=int, default=25, help="Finalize and reopen the dataset after this many output episodes.")
    parser.add_argument(
        "--max-new-output-episodes",
        type=int,
        default=0,
        help="Commit at most this many new output episodes, then exit successfully for a clean process-level resume (0 means no limit).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted conversion in an existing output directory.")
    args = parser.parse_args()
    root = args.input_root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    tolerance_ns = int(args.tolerance_ms * 1_000_000)
    plans = [scan_episode(path, tolerance_ns, args.min_frames) for path in sorted(root.iterdir()) if path.is_dir()]
    ready = [plan for plan in plans if plan["status"] == "ready"]
    rejected = [{key: value for key, value in plan.items() if key not in {"left_paths", "right_paths", "mappings", "runs"}} for plan in plans if plan["status"] != "ready"]
    summary = {
        "source_root": str(root), "task": TASK, "tolerance_ms": args.tolerance_ms, "min_frames": args.min_frames,
        "input_episodes": len(plans), "ready_input_episodes": len(ready), "rejected_input_episodes": len(rejected),
        "output_episodes": sum(len(plan["runs"]) for plan in ready), "output_frames": sum(plan["selected_frames"] for plan in ready),
        "rejected": rejected,
        "limitations": ["Timestamp-nearest matching is inferred alignment, not an official aligned_joints product.", "Arm action and right-gripper command semantics require robot-side validation."],
    }
    if args.mode == "plan":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return

    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    if args.checkpoint_episodes < 1:
        raise ValueError("--checkpoint-episodes must be positive")
    if args.max_new_output_episodes < 0:
        raise ValueError("--max-new-output-episodes must be non-negative")
    # The dataset class resolves this helper from its module at save time.
    lerobot_dataset_module._encode_video_worker = encode_h264_episode
    # LeRobot v3 writes global stats after every episode. In this environment
    # its JSON serializer is intermittently patched by an imported dependency,
    # while the frame/video commits themselves are valid. Stats are recomputed
    # once from the complete sealed dataset after conversion.
    lerobot_dataset_module.write_stats = lambda *_args, **_kwargs: None
    if args.resume:
        recovery = recover_interrupted_output(args.output)
        print(f"Recovered output prefix: {json.dumps(recovery)}", flush=True)
        dataset = LeRobotDataset(repo_id=args.output.name, root=args.output, video_backend="pyav")
        start_episode = dataset.meta.total_episodes
        if start_episode > summary["output_episodes"]:
            raise RuntimeError(f"Output has {start_episode} episodes but the plan only has {summary['output_episodes']}")
        # An interrupted save can leave unsaved PNGs for one or more future
        # episodes. Metadata is the commit record, so staging directories at or
        # after its first uncommitted episode are safe to discard.
        for video_key in dataset.meta.video_keys:
            image_root = (args.output / DEFAULT_IMAGE_PATH.format(
                image_key=video_key, episode_index=0, frame_index=0
            )).parent.parent
            for stale_dir in image_root.glob("episode-*"):
                try:
                    stale_episode = int(stale_dir.name.removeprefix("episode-"))
                except ValueError:
                    continue
                if stale_episode >= start_episode:
                    shutil.rmtree(stale_dir)
    else:
        dataset = LeRobotDataset.create(repo_id=args.output.name, root=args.output, fps=FPS, features=features(), robot_type="a2d_bimanual_arm", use_videos=True, image_writer_threads=16, video_backend="pyav")
        start_episode = 0
    converted = []
    last_checkpoint = 0
    reached_batch_limit = False
    try:
        planned_before = 0
        for index, plan in enumerate(ready, start=1):
            planned_after = planned_before + len(plan["runs"])
            if planned_after <= start_episode:
                planned_before = planned_after
                continue
            start_run = max(0, start_episode - planned_before)
            pending_runs = plan["runs"][start_run:]
            if args.max_new_output_episodes:
                remaining = args.max_new_output_episodes - len(converted)
                if remaining <= 0:
                    reached_batch_limit = True
                    break
                pending_runs = pending_runs[:remaining]
            pending_plan = {**plan, "runs": pending_runs}
            print(f"[{index}/{len(ready)}] {plan['id']}: {pending_plan['segments'][start_run:]}", flush=True)
            converted.extend(convert_episode(dataset, root / plan["id"], pending_plan))
            if len(converted) - last_checkpoint >= args.checkpoint_episodes:
                dataset.finalize()
                dataset = LeRobotDataset(repo_id=args.output.name, root=args.output, video_backend="pyav")
                last_checkpoint = len(converted)
                print(f"Checkpointed {dataset.meta.total_episodes} output episodes", flush=True)
            planned_before = planned_after
            if args.max_new_output_episodes and len(converted) >= args.max_new_output_episodes:
                reached_batch_limit = True
                break
    finally:
        dataset.finalize()
    if reached_batch_limit:
        print(json.dumps({"checkpointed_output_episodes": len(converted), "checkpointed_frames": sum(item["frames"] for item in converted)}, indent=2), flush=True)
        # LeRobot/OpenCV leave native worker threads alive on this host after
        # a long conversion.  finalize() has already committed data and
        # videos; exiting the whole process avoids an interpreter-shutdown
        # deadlock before the next --resume batch.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    summary["resumed_from_output_episode"] = start_episode
    summary["converted_after_resume"] = converted
    (args.output / "conversion_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"converted_output_episodes": len(converted), "converted_frames": sum(item["frames"] for item in converted)}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
