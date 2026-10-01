"""Evaluate four checkpoints from first gripper closure to episode end.

Offline teacher-forced evaluation only. Recorded observations are refreshed at
each model-native chunk boundary. Existing 200-frame predictions are reused.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from evaluate_four_models_postclose import PROMPTS, ROOT, SPECS, episodes, labels, sha
from evaluate_label_conditioned_actions import action_from_model, load_policy
from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.utils import FeatureTransform

OUT = ROOT / "lingbot-vla/output/separated_model/postclose_full_tail_eval_20260930"
OLD_OUT = ROOT / "lingbot-vla/output/separated_model/postclose_eval_20260930"


def prepare(name: str, dataset: Path):
    labs = labels(dataset)
    jobs, truths, anchors, excluded = [], [], [], []
    for meta in episodes(dataset):
        episode = int(meta["episode_index"])
        label = labs[episode]
        if bool(label["had_regrasp"]):
            excluded.append({"episode": episode, "reason": "had_regrasp"})
            continue
        path = dataset / (
            f"data/chunk-{int(meta['data/chunk_index']):03d}/"
            f"file-{int(meta['data/file_index']):03d}.parquet"
        )
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "index", "action"]
        ).to_pydict()
        mask = np.asarray(table["episode_index"]) == episode
        actions = np.asarray(table["action"], dtype=np.float64)[mask]
        indices = np.asarray(table["index"], dtype=np.int64)[mask]
        frames = np.asarray(table["frame_index"], dtype=np.int64)[mask]
        if len(actions) != int(meta["length"]):
            raise ValueError(f"{name}: episode {episode} length mismatch")
        np.testing.assert_array_equal(frames, np.arange(len(actions)))
        closed = actions[:, 15] > 0.5
        crossing = np.flatnonzero(np.diff(closed.astype(np.int8)) == 1) + 1
        if closed[0] or not len(crossing):
            excluded.append(
                {"episode": episode, "reason": "no bounded first closure crossing"}
            )
            continue
        close = int(crossing[0])
        jobs.append(
            {
                "episode": episode,
                "label_up": bool(label["grasp_label_up"]),
                "closure_frame": close,
                "global_index": int(indices[close]),
                "tail_frames": len(actions) - close,
            }
        )
        truths.append(actions[close:])
        anchors.append(actions[close])
    if not jobs:
        raise RuntimeError(f"{name}: no eligible episodes")
    return jobs, truths, np.asarray(anchors), excluded


def make_batch(ds, transform, raw, prompt: str, chunk: int):
    obs = {
        "observation.state": raw["observation.state"].numpy(),
        "task": prompt,
    }
    for key in ds.meta.camera_keys:
        image = (
            raw[key].permute(1, 2, 0).numpy() * 255
        ).round().clip(0, 255).astype(np.uint8)
        image = np.asarray(
            Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR)
        )
        obs[key] = torch.from_numpy(
            np.array(image.transpose(2, 0, 1) / 255.0, copy=True)
        )
    obs["observation.state"] = torch.from_numpy(obs["observation.state"].copy())
    obs["action"] = torch.zeros(chunk, 16)
    obs["action_is_pad"] = torch.zeros(chunk)
    batch = transform.apply(obs)
    batch["state"] = batch["state"].to(torch.bfloat16).float()
    return batch


def infer(name, checkpoint, dataset, chunk, jobs, truths, anchors, excluded):
    cfg = yaml.safe_load((checkpoint / "lingbotvla_cli.yaml").read_text())
    policy, processor, config = load_policy(checkpoint, cfg)
    config.chunk_size = chunk
    if int(config.n_action_steps) != chunk:
        raise ValueError(f"{name}: checkpoint n_action_steps != {chunk}")
    data = SimpleNamespace(**cfg["data"])
    for key in (
        "max_state_dim",
        "max_action_dim",
        "resize_imgs_with_padding",
        "tokenizer_max_length",
    ):
        setattr(data, key, getattr(config, key))
    transform = FeatureTransform(
        Path(data.robot_config_root) / f"{data.data_name}.yaml",
        data,
        processor.tokenizer,
        processor.image_processor,
        chunk_size=chunk,
        norm_stats_path=data.norm_stats_file,
    )
    wrapper = SimpleNamespace(feature_transform=transform)
    ds = LeRobotDataset(str(dataset), image_transforms=None, delta_timestamps=None)
    ds.hf_dataset = ds.hf_dataset.sort("index")
    out = OUT / name
    samples = out / "offline_predictions"
    samples.mkdir(parents=True, exist_ok=True)
    old_samples = OLD_OUT / name / "offline_predictions"
    old_chunks = 200 // chunk
    seed_stride = old_chunks
    total = sum(math.ceil(j["tail_frames"] / chunk) for j in jobs)
    done = reused = 0
    started = time.monotonic()
    predictions, latencies = [], []
    for job_index, (job, truth) in enumerate(zip(jobs, truths)):
        episode_chunks = []
        number_chunks = math.ceil(job["tail_frames"] / chunk)
        for chunk_index in range(number_chunks):
            filename = f"episode-{job['episode']:06d}-chunk-{chunk_index}.npz"
            current = samples / filename
            old = old_samples / filename
            seed = 992100 + seed_stride * job_index + chunk_index
            if chunk_index < old_chunks and old.exists():
                with np.load(old) as saved:
                    prediction = saved["action"]
                    latency = float(saved["infer_ms"])
                reused += 1
            elif current.exists():
                with np.load(current) as saved:
                    cached_seed = int(saved["seed"]) if "seed" in saved else -1
                    if cached_seed == seed:
                        prediction = saved["action"]
                        latency = float(saved["infer_ms"])
                    else:
                        prediction = None
                if prediction is None:
                    index = job["global_index"] + chunk_index * chunk
                    raw = ds[index]
                    if int(raw["episode_index"]) != job["episode"]:
                        raise ValueError(f"{name}: episode boundary crossed")
                    batch = make_batch(
                        ds, transform, raw, PROMPTS[job["label_up"]], chunk
                    )
                    tick = time.monotonic()
                    prediction = action_from_model(policy, wrapper, batch, seed, 10)
                    latency = (time.monotonic() - tick) * 1000
                    if prediction.shape != (chunk, 16) or not np.isfinite(
                        prediction
                    ).all():
                        raise ValueError(f"{name}: bad prediction {prediction.shape}")
                    np.savez_compressed(
                        current, action=prediction, infer_ms=latency, seed=seed
                    )
            else:
                index = job["global_index"] + chunk_index * chunk
                raw = ds[index]
                if int(raw["episode_index"]) != job["episode"]:
                    raise ValueError(f"{name}: episode boundary crossed")
                batch = make_batch(
                    ds, transform, raw, PROMPTS[job["label_up"]], chunk
                )
                tick = time.monotonic()
                prediction = action_from_model(
                    policy,
                    wrapper,
                    batch,
                    seed,
                    10,
                )
                latency = (time.monotonic() - tick) * 1000
                if prediction.shape != (chunk, 16) or not np.isfinite(
                    prediction
                ).all():
                    raise ValueError(f"{name}: bad prediction {prediction.shape}")
                np.savez_compressed(current, action=prediction, infer_ms=latency, seed=seed)
            episode_chunks.append(prediction)
            latencies.append(latency)
            done += 1
            if done % 64 == 0:
                elapsed = time.monotonic() - started
                eta = (total - done) * elapsed / done
                print(
                    f"{name}: {done}/{total}, reused={reused}, ETA={eta:.1f}s",
                    flush=True,
                )
        predictions.append(np.concatenate(episode_chunks)[: len(truth)])

    lengths = np.asarray([len(x) for x in truths], dtype=np.int32)
    max_length = int(lengths.max())
    prediction_pad = np.full((len(jobs), max_length, 16), np.nan)
    truth_pad = np.full_like(prediction_pad, np.nan)
    for index, (prediction, truth) in enumerate(zip(predictions, truths)):
        prediction_pad[index, : len(truth)] = prediction
        truth_pad[index, : len(truth)] = truth
    label_up = np.asarray([j["label_up"] for j in jobs], dtype=bool)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "paired_trajectories.npz",
        prediction=prediction_pad,
        truth=truth_pad,
        anchor=anchors,
        label_up=label_up,
        lengths=lengths,
    )
    error = np.rad2deg(
        prediction_pad[:, :, 7:14] - truth_pad[:, :, 7:14]
    )
    metrics = {
        "model": name,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "norm_stats_file": data.norm_stats_file,
        "norm_stats_sha256": sha(Path(data.norm_stats_file)),
        "chunk_size": chunk,
        "episodes": len(jobs),
        "tail_frames_min": int(lengths.min()),
        "tail_frames_median": float(np.median(lengths)),
        "tail_frames_max": max_length,
        "predictions": total,
        "reused_predictions": reused,
        "right_arm_mae_deg": float(np.nanmean(np.abs(error))),
        "right_arm_rmse_deg": float(np.sqrt(np.nanmean(np.square(error)))),
        "right_arm_mae_by_joint_deg": np.nanmean(
            np.abs(error), axis=(0, 1)
        ).tolist(),
        "inference_ms_mean": float(np.mean(latencies)),
        "groups": {},
    }
    for value, group in ((True, "label_up"), (False, "label_down")):
        mask = label_up == value
        if mask.any():
            metrics["groups"][group] = {
                "episodes": int(mask.sum()),
                "mae_deg": float(np.nanmean(np.abs(error[mask]))),
                "rmse_deg": float(
                    np.sqrt(np.nanmean(np.square(error[mask])))
                ),
            }
    manifest = {
        "model": name,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset),
        "chunk_size": chunk,
        "eligible_episodes": len(jobs),
        "excluded": excluded,
        "jobs": jobs,
        "protocol": (
            "Held-out non-regrasp episodes; first right-gripper closure through "
            "episode end; recorded observations at native chunk boundaries; "
            "teacher-forced offline evaluation; 10 denoising steps."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    del policy, processor, ds
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main():
    os.environ.setdefault(
        "QWEN25_PATH", str(ROOT / "models/Qwen2.5-VL-3B-Instruct-tokenizer")
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (checkpoint, dataset, chunk) in SPECS.items():
        jobs, truths, anchors, excluded = prepare(name, dataset)
        summary[name] = infer(
            name,
            checkpoint,
            dataset,
            chunk,
            jobs,
            truths,
            anchors,
            excluded,
        )
    (OUT / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
