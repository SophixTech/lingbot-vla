"""Evaluate four LingBot-VLA checkpoints on their corresponding test sets.

Offline, teacher-forced protocol: find first right-gripper closure, refresh
recorded image/state at native chunk boundaries, predict 200 frames, compare
right-arm trajectories. No robot connection.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from PIL import Image

from lingbotvla.data.vla_data.base_dataset import LeRobotDataset
from lingbotvla.data.vla_data.utils import FeatureTransform
from evaluate_label_conditioned_actions import action_from_model, load_policy

ROOT = Path("/home/bjtc/Sophix")
REPO = ROOT / "lingbot-vla"
OUT = REPO / "output/separated_model/postclose_eval_20260930"
EVAL_FRAMES = 200
PROMPTS = {
    True: "Pick the soft package from the box and place it on the conveyor belt, keeping the label facing up.",
    False: "Pick the soft package from the box, flip it so the label faces up, and place it on the conveyor belt.",
}
SPECS = {
    "label_up": (REPO / "output/separated_model/label_up/checkpoints/global_step_10000/hf_ckpt", ROOT / "datasets/separated_data/label_up/test", 50),
    "label_down": (REPO / "output/separated_model/label_down/checkpoints/global_step_10000/hf_ckpt", ROOT / "datasets/separated_data/label_down/test", 50),
    "mixed_conditioned": (REPO / "output/separated_model/mixed_conditioned/checkpoints/global_step_10000/hf_ckpt", ROOT / "datasets/marked_9_1_0922/test", 50),
    "mixed_conditioned_chunk25": (REPO / "output/separated_model/mixed_conditioned_chunk25/checkpoints/global_step_10000/hf_ckpt", ROOT / "datasets/marked_9_1_0922/test", 25),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(root: Path) -> dict[int, dict]:
    return {int(x["episode_index"]): x["extra_labels"]["self_annotation"] for x in map(json.loads, (root / "meta/annotations/episode_labels.jsonl").read_text().splitlines())}


def episodes(root: Path) -> list[dict]:
    rows = [r for p in sorted((root / "meta/episodes").glob("**/*.parquet")) for r in pq.read_table(p).to_pylist()]
    return sorted(rows, key=lambda r: int(r["episode_index"]))


def prepare(name: str, dataset: Path, chunk: int):
    labs = labels(dataset)
    jobs, truth, anchor, excluded = [], [], [], []
    for meta in episodes(dataset):
        ep = int(meta["episode_index"]); lab = labs[ep]
        if bool(lab["had_regrasp"]):
            excluded.append({"episode": ep, "reason": "had_regrasp"}); continue
        path = dataset / (
            f"data/chunk-{int(meta['data/chunk_index']):03d}/"
            f"file-{int(meta['data/file_index']):03d}.parquet"
        )
        table = pq.read_table(path, columns=["episode_index", "index", "action"]).to_pydict()
        act = np.asarray(table["action"], dtype=np.float64)
        closed = act[:, 15] > 0.5
        crossing = np.flatnonzero(np.diff(closed.astype(np.int8)) == 1) + 1
        if closed[0] or not len(crossing):
            excluded.append({"episode": ep, "reason": "no bounded first closure crossing"}); continue
        close = int(crossing[0])
        if close + EVAL_FRAMES > len(act):
            excluded.append({"episode": ep, "reason": "less than 200 frames after closure"}); continue
        jobs.append({"episode": ep, "label_up": bool(lab["grasp_label_up"]), "closure_frame": close, "global_index": int(table["index"][close])})
        truth.append(act[close:close + EVAL_FRAMES]); anchor.append(act[close])
    if not jobs: raise RuntimeError(f"{name}: no eligible episodes")
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    return jobs, np.asarray(truth), np.asarray(anchor), excluded


def infer(name: str, checkpoint: Path, dataset_root: Path, chunk: int, jobs, truth, anchor, excluded):
    cfg = yaml.safe_load((checkpoint / "lingbotvla_cli.yaml").read_text())
    policy, processor, config = load_policy(checkpoint, cfg)
    config.chunk_size = chunk
    if int(config.n_action_steps) != chunk: raise ValueError(f"{name}: checkpoint n_action_steps != {chunk}")
    dc = SimpleNamespace(**cfg["data"])
    for key in ("max_state_dim", "max_action_dim", "resize_imgs_with_padding", "tokenizer_max_length"): setattr(dc, key, getattr(config, key))
    transform = FeatureTransform(Path(dc.robot_config_root) / f"{dc.data_name}.yaml", dc, processor.tokenizer, processor.image_processor, chunk_size=chunk, norm_stats_path=dc.norm_stats_file)
    wrapper = SimpleNamespace(feature_transform=transform)
    ds = LeRobotDataset(str(dataset_root), image_transforms=None, delta_timestamps=None); ds.hf_dataset = ds.hf_dataset.sort("index")
    out = OUT / name; sample_dir = out / "offline_predictions"; sample_dir.mkdir(exist_ok=True)
    n_chunks = EVAL_FRAMES // chunk; prediction, latencies = [], []
    for ji, job in enumerate(jobs):
        chunks = []
        for ci in range(n_chunks):
            fp = sample_dir / f"episode-{job['episode']:06d}-chunk-{ci}.npz"
            if fp.exists():
                z = np.load(fp); pred = z["action"]; latency = float(z["infer_ms"])
            else:
                raw = ds[job["global_index"] + ci * chunk]
                if int(raw["episode_index"]) != job["episode"]: raise ValueError(f"{name}: episode boundary crossed")
                obs = {"observation.state": raw["observation.state"].numpy(), "task": PROMPTS[job["label_up"]]}
                for key in ds.meta.camera_keys:
                    image = (raw[key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                    image = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
                    obs[key] = torch.from_numpy(np.array(image.transpose(2, 0, 1) / 255., copy=True))
                obs["observation.state"] = torch.from_numpy(obs["observation.state"].copy()); obs["action"] = torch.zeros(chunk, 16); obs["action_is_pad"] = torch.zeros(chunk)
                batch = transform.apply(obs); batch["state"] = batch["state"].to(torch.bfloat16).float()
                start = time.monotonic(); pred = action_from_model(policy, wrapper, batch, 992100 + n_chunks * ji + ci, 10); latency = (time.monotonic() - start) * 1000
                if pred.shape != (chunk, 16) or not np.isfinite(pred).all(): raise ValueError(f"{name}: bad prediction {pred.shape}")
                np.savez_compressed(fp, action=pred, infer_ms=latency)
            chunks.append(pred); latencies.append(latency)
        prediction.append(np.concatenate(chunks))
    prediction = np.asarray(prediction); labels_arr = np.asarray([j["label_up"] for j in jobs], dtype=bool)
    np.savez_compressed(out / "paired_trajectories.npz", prediction=prediction, truth=truth, anchor=anchor, label_up=labels_arr)
    error = np.rad2deg(prediction[:, :, 7:14] - truth[:, :, 7:14])
    metrics = {"model": name, "checkpoint": str(checkpoint), "dataset": str(dataset_root), "norm_stats_file": dc.norm_stats_file, "norm_stats_sha256": sha(Path(dc.norm_stats_file)), "chunk_size": chunk, "eval_frames": EVAL_FRAMES, "episodes": len(jobs), "predictions": len(jobs) * n_chunks, "right_arm_mae_deg": float(np.abs(error).mean()), "right_arm_rmse_deg": float(np.sqrt(np.square(error).mean())), "right_arm_mae_by_joint_deg": np.abs(error).mean((0, 1)).tolist(), "right_arm_mae_by_chunk_deg": np.abs(error).reshape(len(jobs), n_chunks, chunk, 7).mean((0, 2, 3)).tolist(), "inference_ms_mean": float(np.mean(latencies)), "groups": {}}
    for value, group in ((True, "label_up"), (False, "label_down")):
        mask = labels_arr == value
        if mask.any(): metrics["groups"][group] = {"episodes": int(mask.sum()), "mae_deg": float(np.abs(error[mask]).mean()), "rmse_deg": float(np.sqrt(np.square(error[mask]).mean()))}
    (out / "manifest.json").write_text(json.dumps({"model": name, "checkpoint": str(checkpoint), "dataset": str(dataset_root), "chunk_size": chunk, "eval_frames": EVAL_FRAMES, "eligible_episodes": len(jobs), "excluded": excluded, "jobs": jobs, "protocol": "Held-out non-regrasp episodes; first right-gripper closure; recorded observations at native chunk boundaries; 200-frame teacher-forced offline evaluation; 10 denoising steps."}, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    del policy, processor, ds; gc.collect(); torch.cuda.empty_cache()
    return metrics


def main():
    os.environ.setdefault("QWEN25_PATH", str(ROOT / "models/Qwen2.5-VL-3B-Instruct-tokenizer")); OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (checkpoint, dataset, chunk) in SPECS.items():
        jobs, truth, anchor, excluded = prepare(name, dataset, chunk)
        summary[name] = infer(name, checkpoint, dataset, chunk, jobs, truth, anchor, excluded)
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
