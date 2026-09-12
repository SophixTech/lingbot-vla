#!/usr/bin/env python3
"""Summarize saved offline VLA open-loop rollout predictions.

This script never loads a model or modifies the dataset/checkpoint.  It only
aggregates ``*.npz`` prediction pairs produced by ``open_loop_eval.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIM_NAMES = [
    *[f"left_joint_{index}" for index in range(1, 8)],
    *[f"right_joint_{index}" for index in range(1, 8)],
    "left_gripper",
    "right_gripper",
]


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", default="global_step_10000")
    parser.add_argument("--in-sample", action="store_true")
    args = parser.parse_args()

    files = sorted(args.output.glob("*.npz"), key=lambda path: int(path.stem))
    if not files:
        raise SystemExit(f"No rollout .npz files under {args.output}")

    ground_truth, predictions, per_episode = [], [], []
    for path in files:
        values = np.load(path)
        gt = np.asarray(values["gt_actions"], dtype=np.float64)
        pred = np.asarray(values["predicted_actions"], dtype=np.float64)
        error = pred - gt
        ground_truth.append(gt)
        predictions.append(pred)
        per_episode.append(
            {
                "episode_index": int(path.stem),
                "samples": len(gt),
                "mae": float(np.abs(error).mean()),
                "mse": float((error**2).mean()),
                "max_abs_error": float(np.abs(error).max()),
                "left_gripper_mae": float(np.abs(error[:, -2]).mean()),
                "right_gripper_mae": float(np.abs(error[:, -1]).mean()),
                "left_gripper_accuracy": float(
                    ((gt[:, -2] >= 0.5) == (pred[:, -2] >= 0.5)).mean()
                ),
                "right_gripper_accuracy": float(
                    ((gt[:, -1] >= 0.5) == (pred[:, -1] >= 0.5)).mean()
                ),
                "right_gripper_gt_open_rate": float((gt[:, -1] >= 0.5).mean()),
                "right_gripper_pred_open_rate": float((pred[:, -1] >= 0.5).mean()),
            }
        )

    gt = np.concatenate(ground_truth)
    pred = np.concatenate(predictions)
    error = pred - gt
    write_tsv(args.output / "per_episode_detailed.tsv", per_episode)

    dimension_rows = []
    for index, name in enumerate(DIM_NAMES):
        row = {
            "dimension": index,
            "name": name,
            "mae": float(np.abs(error[:, index]).mean()),
            "mse": float((error[:, index] ** 2).mean()),
            "max_abs_error": float(np.abs(error[:, index]).max()),
        }
        if index >= 14:
            row.update(
                {
                    "state_accuracy": float(
                        ((gt[:, index] >= 0.5) == (pred[:, index] >= 0.5)).mean()
                    ),
                    "gt_open_rate": float((gt[:, index] >= 0.5).mean()),
                    "pred_open_rate": float((pred[:, index] >= 0.5).mean()),
                }
            )
        dimension_rows.append(row)
    write_tsv(args.output / "dimension_metrics.tsv", dimension_rows)

    horizon_rows = []
    for index in range(16):
        horizon_gt = np.concatenate([array[index::16] for array in ground_truth])
        horizon_pred = np.concatenate([array[index::16] for array in predictions])
        horizon_error = horizon_pred - horizon_gt
        horizon_rows.append(
            {
                "horizon": index + 1,
                "samples": len(horizon_gt),
                "mae": float(np.abs(horizon_error).mean()),
                "mse": float((horizon_error**2).mean()),
                "right_gripper_mae": float(np.abs(horizon_error[:, -1]).mean()),
                "right_gripper_accuracy": float(
                    ((horizon_gt[:, -1] >= 0.5) == (horizon_pred[:, -1] >= 0.5)).mean()
                ),
            }
        )
    write_tsv(args.output / "horizon_metrics.tsv", horizon_rows)

    ranked = sorted(per_episode, key=lambda row: row["mae"])
    summary = {
        "checkpoint": args.checkpoint,
        "protocol": "offline open-loop replay with recorded observations; predictions do not alter later observations; not physical robot rollout",
        "evaluation_kind": "in-sample replay" if args.in_sample else "held-out replay",
        "episodes": len(per_episode),
        "samples": len(gt),
        "action_mae": float(np.abs(error).mean()),
        "action_mse": float((error**2).mean()),
        "max_abs_error": float(np.abs(error).max()),
        "best_episode": ranked[0]["episode_index"],
        "worst_episode": ranked[-1]["episode_index"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([str(row["episode_index"]) for row in per_episode], [row["mae"] for row in per_episode])
    axes[0].set(title="Full-episode action MAE", xlabel="episode", ylabel="MAE")
    axes[1].plot([row["horizon"] for row in horizon_rows], [row["mae"] for row in horizon_rows], "o-", label="all actions")
    axes[1].plot([row["horizon"] for row in horizon_rows], [row["right_gripper_mae"] for row in horizon_rows], "s-", label="right gripper")
    axes[1].set(title="Error by 16-step action horizon", xlabel="horizon", ylabel="MAE")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(args.output / "rollout_summary.png", dpi=180)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
