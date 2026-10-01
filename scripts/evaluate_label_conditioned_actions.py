"""Compare marked_full0919 action trajectories for label-up/down conditions.

This is an offline, in-dataset evaluation.  It reports both:
1) natural label-up vs label-down episode groups; and
2) paired counterfactual prompts on identical image/state observations.

All reported arm values are absolute joint targets after FeatureTransform.unapply,
converted from radians to degrees.  It never connects to a robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from safetensors import safe_open
from transformers import AutoConfig

from lerobot.configs.policies import PreTrainedConfig
from deploy.lingbot_vla_policy import merge_qwen_config, set_seed_everywhere
from lingbotvla.models import build_processor
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.utils.lora_utils import add_lora_to_model
from lingbotvla.data.vla_data.base_dataset import VLADataset


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output/marked_full0919")
DEFAULT_CK = ROOT / "checkpoints/global_step_10000/hf_ckpt"
UP_PROMPT = "Pick the soft package from the box and place it on the conveyor belt, keeping the label facing up."
DOWN_PROMPT = "Pick the soft package from the box, flip it so the label faces up, and place it on the conveyor belt."


def load_policy(checkpoint: Path, cfg: dict):
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.__dict__.update({k: v for k, v in {**cfg["model"], **cfg["train"]}.items() if not hasattr(config, k)})
    config.attention_implementation = "eager"
    tokenizer_path = os.environ["QWEN25_PATH"]
    config.tokenizer_path = tokenizer_path
    config = merge_qwen_config(config, AutoConfig.from_pretrained(tokenizer_path))
    config.use_cache = True
    processor = build_processor(tokenizer_path)
    policy = LingbotVlaPolicy(config, tokenizer_path=tokenizer_path)
    policy = add_lora_to_model(
        policy,
        lora_rank=cfg["train"]["lora_rank"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_target_modules=cfg["train"]["lora_target_modules"],
        lora_target_modules_support=("q_proj", "k_proj", "v_proj", "o_proj"),
    )
    weights = {}
    for p in checkpoint.glob("*.safetensors"):
        with safe_open(p, framework="pt", device="cpu") as f:
            weights.update({k: f.get_tensor(k) for k in f.keys()})
    policy.load_state_dict(weights, strict=True)
    del weights
    policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()
    return policy, processor, config


def episode_groups(ds):
    groups = {True: [], False: []}
    for ep, record in ds.episode_labels.items():
        # The two rare stress trajectories have unusual left-arm excursions and
        # are excluded from the main comparison; they are not label effects.
        if ep in (806, 865):
            continue
        label = bool(record["extra_labels"]["self_annotation"]["grasp_label_up"])
        groups[label].append(int(ep))
    return {k: sorted(v) for k, v in groups.items()}


def action_from_model(policy, ds, batch, seed: int, denoising_steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        args = [
            batch[k].unsqueeze(0).to(device="cuda", dtype=torch.bfloat16 if k in ("images", "state") else batch[k].dtype)
            for k in ("images", "img_masks", "lang_tokens", "lang_masks", "state")
        ]
        pred = policy.model.sample_actions(*args, num_steps=denoising_steps).squeeze(0).float().cpu()
    out = ds.feature_transform.unapply({**batch, "actions": pred.clone()})["action"].numpy()
    return out.astype(np.float64)


def batch_with_prompt(ds, idx: int, prompt: str):
    raw = ds.dataset[idx]
    raw["task"] = prompt
    return ds.feature_transform.apply(raw)


def ci95(values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(9919)
    # Keep the bootstrap compact; values are sample-level (not frame-level).
    reps = 2000
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_natural(natural: dict[str, list[np.ndarray]]):
    out = {}
    for group, arrs in natural.items():
        x = np.stack(arrs, axis=0)  # episodes/anchors, horizon, action_dim
        right_deg = np.rad2deg(x[:, :, 7:14])
        out[group] = {
            "samples": int(len(x)),
            "right_arm_mean_deg": right_deg.mean(axis=0).tolist(),
            "right_arm_std_deg": right_deg.std(axis=0).tolist(),
            "right_arm_mean_final_minus_first_deg": (right_deg[:, -1] - right_deg[:, 0]).mean(axis=0).tolist(),
            "right_arm_rms_path_deg": np.sqrt(np.mean(np.square(np.diff(right_deg, axis=1)), axis=(1, 2))).mean().item(),
            "right_arm_mean_abs_from_first_deg": np.mean(np.abs(right_deg - right_deg[:, :1]), axis=(0, 1)).tolist(),
            "gripper_mean": x[:, :, 14:16].mean(axis=(0, 1)).tolist(),
            "gripper_std": x[:, :, 14:16].std(axis=(0, 1)).tolist(),
        }
    up = np.stack(natural["label_up"], axis=0)
    down = np.stack(natural["label_down"], axis=0)
    # Episode/anchor-level aggregate difference, with a conservative bootstrap CI.
    up_r = np.rad2deg(up[:, :, 7:14]).mean(axis=1)
    down_r = np.rad2deg(down[:, :, 7:14]).mean(axis=1)
    diff = up_r.mean(axis=0) - down_r.mean(axis=0)
    out["up_minus_down_episode_mean"] = {
        "mean_difference_deg": diff.tolist(),
        "ci95_difference_deg": [ci95((up_r[:, j][:, None] - down_r[: len(up_r), j][:, None]).ravel()) for j in range(7)]
        if len(up_r) == len(down_r) else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CK)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "label_conditioned_action_eval_20260921")
    ap.add_argument("--episodes-per-label", type=int, default=32)
    ap.add_argument("--denoising-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=9919)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((args.checkpoint / "lingbotvla_cli.yaml").read_text())
    set_seed_everywhere(42)
    policy, processor, config = load_policy(args.checkpoint, cfg)
    data = SimpleNamespace(**cfg["data"])
    data.exclude_regrasp = False
    ds = VLADataset(data.train_path, data.data_name, data, "configs/robot_configs", config=config,
                    tokenizer=processor.tokenizer, image_processor=processor.image_processor)
    assert len(ds) == 748377, len(ds)
    groups = episode_groups(ds)
    rng = np.random.default_rng(args.seed)
    n = min(args.episodes_per_label, len(groups[True]), len(groups[False]))
    selected = {True: sorted(rng.choice(groups[True], n, replace=False).tolist()),
                False: sorted(rng.choice(groups[False], n, replace=False).tolist())}
    anchors = (0.2, 0.5, 0.8)
    natural = {"label_up": [], "label_down": []}
    cf_up, cf_down, cf_meta = [], [], []
    records = []
    total = n * 2 * len(anchors)
    done = 0
    for label in (True, False):
        for ep in selected[label]:
            meta = ds.dataset.meta.episodes[ep]
            for frac in anchors:
                frame = int((meta["length"] - 1) * frac)
                idx = int(meta["dataset_from_index"]) + frame
                batch_actual = ds.getdata(idx)
                actual = action_from_model(policy, ds, batch_actual, args.seed + done, args.denoising_steps)
                natural["label_up" if label else "label_down"].append(actual)
                # Same visual/state, same diffusion seed, only the language condition changes.
                batch_up = batch_with_prompt(ds, idx, UP_PROMPT)
                batch_down = batch_with_prompt(ds, idx, DOWN_PROMPT)
                pred_up = action_from_model(policy, ds, batch_up, 100000 + args.seed + done, args.denoising_steps)
                pred_down = action_from_model(policy, ds, batch_down, 100000 + args.seed + done, args.denoising_steps)
                cf_up.append(pred_up)
                cf_down.append(pred_down)
                cf_meta.append({"episode": ep, "label_up": label, "frame": frame, "fraction": frac, "index": idx})
                records.append({"episode": ep, "label_up": label, "frame": frame, "fraction": frac, "index": idx})
                done += 1
                if done % 12 == 0:
                    print(f"evaluated {done}/{total}", flush=True)

    natural_np = {k: np.stack(v, axis=0) for k, v in natural.items()}
    cf_up_np, cf_down_np = np.stack(cf_up, axis=0), np.stack(cf_down, axis=0)
    cf_diff_deg = np.rad2deg(cf_up_np[:, :, 7:14] - cf_down_np[:, :, 7:14])
    cf_abs = np.abs(cf_diff_deg)
    # Per-sample trajectory summaries, retaining sample-level independence.
    per_sample = {
        "right_arm_mean_abs_difference_deg": cf_abs.mean(axis=(1, 2)),
        "right_arm_rms_difference_deg": np.sqrt(np.mean(np.square(cf_diff_deg), axis=(1, 2))),
        "right_arm_max_abs_difference_deg": cf_abs.max(axis=(1, 2)),
        "right_arm_final_mean_abs_difference_deg": cf_abs[:, -1].mean(axis=1),
    }
    summaries = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": hashlib.sha256((args.checkpoint / "lingbotvla_cli.yaml").read_bytes()).hexdigest(),
        "norm_stats": data.norm_stats_file,
        "norm_stats_sha256": hashlib.sha256(Path(data.norm_stats_file).read_bytes()).hexdigest(),
        "dataset": data.train_path,
        "denoising_steps": args.denoising_steps,
        "protocol": "Balanced 32 episodes per label (excluding stress episodes 806/865), three anchors at 20/50/80 percent; 96 natural samples per label. Counterfactual predictions use identical image/state and identical diffusion seed, switching only the exact training prompt.",
        "population": {"label_up": len(groups[True]), "label_down": len(groups[False])},
        "selected_episodes": {"label_up": selected[True], "label_down": selected[False]},
        "natural": summarize_natural(natural),
        "counterfactual": {
            "samples": int(len(cf_diff_deg)),
            "right_arm_mean_difference_up_minus_down_deg_by_horizon_joint": cf_diff_deg.mean(axis=0).tolist(),
            "right_arm_mean_abs_difference_deg": float(cf_abs.mean()),
            "right_arm_rms_difference_deg": float(np.sqrt(np.mean(np.square(cf_diff_deg)))),
            "right_arm_max_abs_difference_deg": float(cf_abs.max()),
            "right_arm_final_mean_abs_difference_deg": float(cf_abs[:, -1].mean()),
            "mean_abs_by_joint_deg": cf_abs.mean(axis=(0, 1)).tolist(),
            "p95_abs_by_joint_deg": np.quantile(cf_abs, 0.95, axis=(0, 1)).tolist(),
            "ci95_mean_abs_difference_deg": ci95(cf_abs.mean(axis=(1, 2))),
            "fraction_samples_mean_abs_gt_1deg": float(np.mean(per_sample["right_arm_mean_abs_difference_deg"] > 1.0)),
            "fraction_samples_mean_abs_gt_2deg": float(np.mean(per_sample["right_arm_mean_abs_difference_deg"] > 2.0)),
            "fraction_joint_time_abs_gt_1deg": float(np.mean(cf_abs > 1.0)),
            "fraction_joint_time_abs_gt_2deg": float(np.mean(cf_abs > 2.0)),
        },
    }
    (out / "metrics.json").write_text(json.dumps(summaries, indent=2))
    (out / "samples.json").write_text(json.dumps(records, indent=2))
    (out / "counterfactual_meta.json").write_text(json.dumps(cf_meta, indent=2))
    np.savez_compressed(out / "trajectories.npz", label_up=natural_np["label_up"], label_down=natural_np["label_down"],
                        counterfactual_up=cf_up_np, counterfactual_down=cf_down_np, counterfactual_diff_deg=cf_diff_deg)
    print(json.dumps(summaries["counterfactual"], indent=2), flush=True)
    print(json.dumps({"natural": summaries["natural"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
