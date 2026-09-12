#!/usr/bin/env python3
"""Train and evaluate an ACT baseline for the A2D soft-package dataset.

The source LeRobot dataset is read-only. This runner resizes each decoded
camera batch to a common size before ACT consumes it, and reserves complete
episodes for validation so no trajectory crosses the train/validation split.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


CAMERAS = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)


class FrameCachedDataset(LeRobotDataset):
    """Read visual observations from JPEG frames, not a runtime video decoder."""

    def __init__(self, *args, frame_cache: Path | None = None, **kwargs) -> None:
        self.frame_cache = frame_cache
        super().__init__(*args, **kwargs)

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        if self.frame_cache is None:
            return super()._query_videos(query_timestamps, ep_idx)
        ep = self.meta.episodes[ep_idx]
        frames: dict[str, torch.Tensor] = {}
        for video_key, timestamps in query_timestamps.items():
            source = Path(self.meta.get_video_file_path(ep_idx, video_key)).relative_to("videos").with_suffix("")
            base = self.frame_cache / source
            offset_s = float(ep[f"videos/{video_key}/from_timestamp"])
            images = []
            for timestamp in timestamps:
                # ffmpeg numbers exported frames from one; timestamps are at 30 FPS.
                frame_number = int(round((offset_s + float(timestamp)) * self.fps)) + 1
                path = base / f"{frame_number:08d}.jpg"
                with Image.open(path) as image:
                    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                images.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
            stacked = torch.stack(images)
            frames[video_key] = stacked[0] if len(images) == 1 else stacked
        return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="Resume from a checkpoint in an existing output directory.")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--loader-timeout", type=float, default=120.0,
                        help="Seconds to wait for a DataLoader batch when workers are enabled.")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Batches prefetched by each DataLoader worker.")
    parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default="torchcodec")
    parser.add_argument("--frame-cache", type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def make_split(total_episodes: int) -> tuple[list[int], list[int]]:
    # Deterministic episode-level split: every tenth full trajectory is held out.
    validation = [episode for episode in range(total_episodes) if episode % 10 == 0]
    training = [episode for episode in range(total_episodes) if episode % 10 != 0]
    return training, validation


def numeric_stats(dataset_root: Path, train_episodes: set[int]) -> dict[str, dict[str, torch.Tensor]]:
    """Compute train-only mean/std stats without decoding image videos."""
    sums = {"observation.state": np.zeros(16, dtype=np.float64), "action": np.zeros(16, dtype=np.float64)}
    squares = {key: np.zeros(16, dtype=np.float64) for key in sums}
    counts = {key: 0 for key in sums}
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "observation.state", "action"])
        episode = np.asarray(table["episode_index"].to_numpy())
        keep = np.isin(episode, list(train_episodes))
        if not keep.any():
            continue
        for key in sums:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)[keep]
            sums[key] += values.sum(axis=0)
            squares[key] += np.square(values).sum(axis=0)
            counts[key] += len(values)
    result: dict[str, dict[str, torch.Tensor]] = {}
    for key in sums:
        mean = sums[key] / counts[key]
        variance = np.maximum(squares[key] / counts[key] - np.square(mean), 1e-12)
        result[key] = {
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(np.sqrt(variance), dtype=torch.float32),
        }
    imagenet_mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32)
    imagenet_std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)
    for camera in CAMERAS:
        result[camera] = {"mean": imagenet_mean, "std": imagenet_std}
    return result


def resize_images(batch: dict[str, object], image_size: int) -> dict[str, object]:
    for camera in CAMERAS:
        images = batch[camera]
        assert isinstance(images, torch.Tensor)
        batch[camera] = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return batch


def make_loader(dataset: LeRobotDataset, episode_ids: list[int], batch_size: int, workers: int,
                chunk_size: int, shuffle: bool, timeout: float, prefetch_factor: int) -> DataLoader:
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=episode_ids,
        drop_n_last_frames=chunk_size - 1,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        # The bounded loader timeout makes video-decoder failures explicit.
        # The ACT cache contains small uniformly shaped video frames, avoiding
        # the high-resolution random-seek pressure seen with the raw export.
        persistent_workers=workers > 0,
        timeout=timeout if workers > 0 else 0,
        prefetch_factor=prefetch_factor if workers > 0 else None,
    )


def evaluate(policy, preprocessor, loader: DataLoader, image_size: int, max_batches: int) -> tuple[float, float]:
    policy.eval()
    losses: list[float] = []
    abs_errors: list[float] = []
    with torch.no_grad():
        for index, raw_batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = preprocessor(resize_images(raw_batch, image_size))
            prediction = policy.predict_action_chunk(batch)
            valid = ~batch["action_is_pad"].unsqueeze(-1)
            error = torch.abs(prediction - batch["action"]) * valid
            abs_errors.append(float(error.sum().item() / valid.sum().item() / 16))
            losses.append(float(error.mean().item()))
    policy.train()
    return float(np.mean(losses)), float(np.mean(abs_errors))


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.resume is None:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this ACT run")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDataset("package_training_data", root=args.dataset, video_backend=args.video_backend).meta
    train_episodes, validation_episodes = make_split(metadata.total_episodes)
    split_path = args.output / "episode_split.json"
    if not (args.resume is not None and split_path.exists()):
        split_path.write_text(json.dumps({
            "seed": args.seed,
            "split_rule": "episode_index % 10 == 0 is validation",
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
        }, indent=2) + "\n")

    delta_timestamps = {"action": [frame / metadata.fps for frame in range(args.chunk_size)]}
    # Keep the full global index table. EpisodeAwareSampler selects the split;
    # filtering the dataset here would make its global action-chunk indices
    # disagree with the sampler at episode boundaries.
    dataset_class = FrameCachedDataset if args.frame_cache else LeRobotDataset
    dataset_kwargs = {"frame_cache": args.frame_cache.resolve()} if args.frame_cache else {}
    train_ds = dataset_class("package_training_data", root=args.dataset,
                             delta_timestamps=delta_timestamps, video_backend=args.video_backend, **dataset_kwargs)
    validation_ds = dataset_class("package_training_data", root=args.dataset,
                                  delta_timestamps=delta_timestamps, video_backend=args.video_backend, **dataset_kwargs)
    stats = numeric_stats(args.dataset, set(train_episodes))

    policy_config = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=1,
        device="cuda",
        use_amp=False,
        push_to_hub=False,
        vision_backbone="resnet18",
        # Keep the run fully offline. The pretrained ResNet weights are an
        # optional initialization and are not present locally on this host.
        pretrained_backbone_weights=None,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=32,
        n_vae_encoder_layers=4,
        kl_weight=10.0,
        optimizer_lr=1e-5,
        optimizer_lr_backbone=1e-5,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )
    policy = make_policy(policy_config, ds_meta=train_ds.meta)
    preprocessor, _ = make_pre_post_processors(policy_config, dataset_stats=stats)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)
    train_loader = make_loader(train_ds, train_episodes, args.batch_size, args.workers,
                               args.chunk_size, shuffle=True, timeout=args.loader_timeout,
                               prefetch_factor=args.prefetch_factor)
    validation_loader = make_loader(validation_ds, validation_episodes, args.batch_size, args.workers,
                                    args.chunk_size, shuffle=False, timeout=args.loader_timeout,
                                    prefetch_factor=args.prefetch_factor)
    train_iter = iter(train_loader)
    metrics_path = args.output / "metrics.jsonl"
    config = vars(args) | {
        "train_episode_count": len(train_episodes), "validation_episode_count": len(validation_episodes),
        "train_frame_count": train_ds.num_frames, "validation_frame_count": validation_ds.num_frames,
        "device": torch.cuda.get_device_name(),
    }
    config_path = args.output / "run_config.json"
    if args.resume is None or not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, default=str) + "\n")

    start_step = 0
    if args.resume is not None:
        checkpoint_path = args.resume.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        start_step = int(checkpoint["step"])
        if start_step >= args.steps:
            raise ValueError(f"Checkpoint step {start_step} is not below target {args.steps}")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(json.dumps({"event": "resumed", "step": start_step,
                          "checkpoint": str(checkpoint_path), **config}, default=str), flush=True)
    else:
        print(json.dumps({"event": "started", **config}, default=str), flush=True)

    policy.train()
    rolling_loss: list[float] = []
    start = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        try:
            raw_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            raw_batch = next(train_iter)
        load_done = time.monotonic()
        batch = preprocessor(resize_images(raw_batch, args.image_size))
        optimizer.zero_grad(set_to_none=True)
        loss, details = policy.forward(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0).item())
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        rolling_loss.append(float(loss.item()))

        if step % args.log_every == 0:
            elapsed = time.monotonic() - start
            record = {"event": "train", "step": step, "loss": float(np.mean(rolling_loss)),
                      "l1_loss": details.get("l1_loss"), "kld_loss": details.get("kld_loss"),
                      "grad_norm": grad_norm, "steps_per_s": step / elapsed,
                      "data_wait_s": load_done - start if step == 1 else None,
                      "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            rolling_loss.clear()

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, val_mae = evaluate(policy, preprocessor, validation_loader, args.image_size, args.eval_batches)
            record = {"event": "validation", "step": step, "loss": val_loss, "action_mae_normalized": val_mae}
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        if step % args.save_every == 0 or step == args.steps:
            checkpoint = args.output / f"checkpoint_{step:06d}.pt"
            torch.save({"step": step, "policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                        "policy_config": policy_config, "stats": stats, "args": vars(args)}, checkpoint)
            print(json.dumps({"event": "checkpoint", "step": step, "path": str(checkpoint)}), flush=True)

    print(json.dumps({"event": "completed", "steps": args.steps, "elapsed_s": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
