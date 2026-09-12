#!/usr/bin/env python3
"""Aggregate offline metrics for the package A2D LingBot-VLA run.

This intentionally operates only on saved predictions and metadata. It does
not claim physical robot success or causal closed-loop performance.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output/package_a2d_lora_4090")
OUT = ROOT / "offline_analysis"
STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
DIM_NAMES = [
    *[f"left_joint_{i}" for i in range(1, 8)],
    *[f"right_joint_{i}" for i in range(1, 8)],
    "left_gripper",
    "right_gripper",
]


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_preds(step: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    folder = ROOT / f"open_loop_holdout_step{step}"
    result = {}
    for p in sorted(folder.glob("*.npz")):
        try:
            eid = int(p.stem)
        except ValueError:
            continue
        z = np.load(p)
        result[eid] = (np.asarray(z["gt_actions"], dtype=np.float64), np.asarray(z["predicted_actions"], dtype=np.float64))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = {step: load_preds(step) for step in STEPS}

    ck_rows = []
    for step, episodes in all_data.items():
        if not episodes:
            continue
        gt = np.concatenate([v[0] for v in episodes.values()])
        pred = np.concatenate([v[1] for v in episodes.values()])
        err = pred - gt
        # Right gripper is the final action dimension; values are normalized
        # continuously and thresholded at zero for open/closed accuracy.
        gtrue, gpred = gt[:, -1], pred[:, -1]
        ck_rows.append({
            "step": step,
            "episodes": len(episodes),
            "samples": len(gt),
            "mae": float(np.mean(np.abs(err))),
            "mse": float(np.mean(err ** 2)),
            "right_gripper_mae": float(np.mean(np.abs(err[:, -1]))),
            "right_gripper_accuracy": float(np.mean((gtrue >= 0.5) == (gpred >= 0.5))),
            "right_gripper_true_open_rate": float(np.mean(gtrue >= 0.5)),
            "right_gripper_pred_open_rate": float(np.mean(gpred >= 0.5)),
            "pred_action_jump_mean": float(np.mean(np.abs(np.diff(pred, axis=0)))),
            "gt_action_jump_mean": float(np.mean(np.abs(np.diff(gt, axis=0)))),
            "max_abs_error": float(np.max(np.abs(err))),
        })
    write_tsv(OUT / "checkpoint_metrics.tsv", ck_rows)

    # Detailed metrics for the final checkpoint.
    final = all_data[10000]
    gt = np.concatenate([v[0] for v in final.values()])
    pred = np.concatenate([v[1] for v in final.values()])
    err = pred - gt
    dim_rows = []
    for i, name in enumerate(DIM_NAMES):
        row = {"dimension": i, "name": name, "mae": float(np.mean(np.abs(err[:, i]))), "mse": float(np.mean(err[:, i] ** 2)), "max_abs_error": float(np.max(np.abs(err[:, i])))}
        if i >= 14:
            row["binary_accuracy"] = float(np.mean((gt[:, i] >= 0.5) == (pred[:, i] >= 0.5)))
            row["true_open_rate"] = float(np.mean(gt[:, i] >= 0.5))
            row["pred_open_rate"] = float(np.mean(pred[:, i] >= 0.5))
        else:
            row["binary_accuracy"] = ""
            row["true_open_rate"] = ""
            row["pred_open_rate"] = ""
        dim_rows.append(row)
    write_tsv(OUT / "step10000_dimension_metrics.tsv", dim_rows)

    horizon_rows = []
    for h in range(16):
        # Predictions are stored as consecutive action chunks per episode;
        # preserve chunk boundaries when selecting a horizon position.
        gparts, pparts = [], []
        for eg, ep in final.values():
            gparts.append(eg[h::16])
            pparts.append(ep[h::16])
        hg, hp = np.concatenate(gparts), np.concatenate(pparts)
        e = hp - hg
        horizon_rows.append({"horizon": h + 1, "samples": len(hg), "mae": float(np.mean(np.abs(e))), "mse": float(np.mean(e ** 2)), "right_gripper_mae": float(np.mean(np.abs(e[:, -1]))), "right_gripper_accuracy": float(np.mean((hg[:, -1] >= 0.5) == (hp[:, -1] >= 0.5)))})
    write_tsv(OUT / "step10000_horizon_metrics.tsv", horizon_rows)

    # Per-episode aggregate and a simple task-level generalization report.
    per_rows = []
    for eid, (eg, ep) in sorted(final.items()):
        ee = ep - eg
        per_rows.append({"episode": eid, "samples": len(eg), "mae": float(np.mean(np.abs(ee))), "mse": float(np.mean(ee ** 2)), "right_gripper_mae": float(np.mean(np.abs(ee[:, -1]))), "right_gripper_accuracy": float(np.mean((eg[:, -1] >= 0.5) == (ep[:, -1] >= 0.5))), "max_abs_error": float(np.max(np.abs(ee)))})
    write_tsv(OUT / "step10000_per_episode_detailed.tsv", per_rows)
    gen_rows = [{"split": "held_out_task", "task": "Pick a soft package from the box, orient its label upward, and place it on the conveyor belt.", "episodes": len(final), "mae": float(np.mean(np.abs(err))), "mse": float(np.mean(err ** 2)), "right_gripper_accuracy": float(np.mean((gt[:, -1] >= 0.5) == (pred[:, -1] >= 0.5))), "object_id": "unavailable in metadata", "scene_id": "unavailable in metadata", "status": "task-level only; object/scene generalization not measurable"}]
    write_tsv(OUT / "generalization_metrics.tsv", gen_rows)

    # Compact static plots for reports and quick inspection.
    plt.style.use("seaborn-v0_8-whitegrid")
    if ck_rows:
        s = np.array([r["step"] for r in ck_rows]);
        fig, ax1 = plt.subplots(figsize=(10, 5)); ax1.plot(s, [r["mae"] for r in ck_rows], "o-", label="MAE"); ax1.plot(s, [r["mse"] for r in ck_rows], "s-", label="MSE"); ax1.set_xlabel("training step"); ax1.set_ylabel("error"); ax1.set_title("Package A2D held-out action error by checkpoint"); ax1.legend(); fig.tight_layout(); fig.savefig(OUT / "checkpoint_mae_mse.png", dpi=180); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(s, [r["right_gripper_accuracy"] for r in ck_rows], "o-", color="#d95f02"); ax.set_ylim(0, 1.05); ax.set_xlabel("training step"); ax.set_ylabel("right gripper sign accuracy"); ax.set_title("Right gripper open/closed accuracy"); fig.tight_layout(); fig.savefig(OUT / "checkpoint_gripper_accuracy.png", dpi=180); plt.close(fig)
    h = np.array([r["horizon"] for r in horizon_rows]); fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(h, [r["mae"] for r in horizon_rows], "o-", label="all action MAE"); ax.plot(h, [r["right_gripper_mae"] for r in horizon_rows], "s-", label="right gripper MAE"); ax.set_xlabel("prediction horizon within 16-step chunk"); ax.set_ylabel("absolute error"); ax.set_title("Error versus prediction horizon (step 10000)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "horizon_error.png", dpi=180); plt.close(fig)
    names = [r["name"] for r in dim_rows]; vals = [r["mae"] for r in dim_rows]; fig, ax = plt.subplots(figsize=(11, 5)); ax.bar(np.arange(len(names)), vals, color=["#4c78a8"] * 14 + ["#f58518", "#e45756"]); ax.set_xticks(np.arange(len(names)), names, rotation=60, ha="right"); ax.set_ylabel("MAE"); ax.set_title("Step 10000 action MAE by dimension"); fig.tight_layout(); fig.savefig(OUT / "dimension_mae.png", dpi=180); plt.close(fig)

    # Two trajectory views: best and worst holdout episode.
    ranked = sorted(per_rows, key=lambda r: r["mae"])
    chosen = [ranked[0]["episode"], ranked[-1]["episode"]]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(12, 6), squeeze=False)
    for row_i, eid in enumerate(chosen):
        eg, ep = final[eid]; t = np.arange(len(eg));
        axes[row_i, 0].plot(t, eg[:, 7], label="GT right joint 1"); axes[row_i, 0].plot(t, ep[:, 7], "--", label="pred right joint 1"); axes[row_i, 0].set_title(f"Episode {eid}: right arm joint 1"); axes[row_i, 0].set_xlabel("sample"); axes[row_i, 0].legend()
        axes[row_i, 1].plot(t, eg[:, 15], label="GT right gripper"); axes[row_i, 1].plot(t, ep[:, 15], "--", label="pred right gripper"); axes[row_i, 1].axhline(0, color="k", lw=.7); axes[row_i, 1].set_title(f"Episode {eid}: right gripper"); axes[row_i, 1].set_xlabel("sample"); axes[row_i, 1].legend()
    fig.suptitle("Step 10000 offline trajectory rollout comparison (open-loop prediction)"); fig.tight_layout(); fig.savefig(OUT / "trajectory_rollout_examples.png", dpi=180); plt.close(fig)

    (OUT / "analysis_summary.json").write_text(json.dumps({"checkpoint_count": len(ck_rows), "final_step": 10000, "final_episodes": len(final), "final_samples": int(len(gt)), "task_count": 1, "object_scene_metadata": False, "selected_trajectory_episodes": chosen, "closed_loop_boundary": "offline replay with recorded observations; not physical robot rollout"}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
