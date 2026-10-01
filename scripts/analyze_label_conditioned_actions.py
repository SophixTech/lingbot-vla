"""Statistical summaries and figures for label-conditioned action evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/label_conditioned_action_eval_20260921")
JOINTS = ["right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_j7"]


def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    q = np.empty_like(ranked)
    running = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[i] * len(ranked) / (i + 1))
        q[i] = running
    out = np.empty_like(flat)
    out[order] = q
    return out.reshape(p.shape)


def main():
    data = np.load(ROOT / "trajectories.npz")
    up = np.rad2deg(data["label_up"][:, :, 7:14])
    down = np.rad2deg(data["label_down"][:, :, 7:14])
    cf_up = np.rad2deg(data["counterfactual_up"][:, :, 7:14])
    cf_down = np.rad2deg(data["counterfactual_down"][:, :, 7:14])
    cf_diff = cf_up - cf_down
    meta = json.loads((ROOT / "samples.json").read_text())

    # Natural comparison: collapse the three anchors within each episode first,
    # so an episode does not receive 3x the weight of another episode.
    def episode_means(arr, label):
        eps = [r["episode"] for r in meta if bool(r["label_up"]) == label]
        unique = sorted(set(eps))
        rows = []
        for ep in unique:
            ix = [i for i, e in enumerate(eps) if e == ep]
            rows.append(arr[ix].mean(axis=0))
        return np.stack(rows), unique

    up_ep, up_ids = episode_means(up, True)
    down_ep, down_ids = episode_means(down, False)
    # Welch test at each horizon x joint, plus Benjamini-Hochberg correction.
    p_nat = np.zeros((50, 7)); t_nat = np.zeros((50, 7)); d_nat = np.zeros((50, 7))
    for h in range(50):
        for j in range(7):
            t, p = stats.ttest_ind(up_ep[:, h, j], down_ep[:, h, j], equal_var=False)
            t_nat[h, j], p_nat[h, j] = t, p
            pooled = np.sqrt((up_ep[:, h, j].var(ddof=1) + down_ep[:, h, j].var(ddof=1)) / 2)
            d_nat[h, j] = (up_ep[:, h, j].mean() - down_ep[:, h, j].mean()) / pooled if pooled > 0 else 0.0
    q_nat = fdr_bh(p_nat)

    # Relative trajectories remove the absolute starting posture, which is
    # important because the two natural label groups do not start from the
    # same robot configuration.
    up_rel, down_rel = up - up[:, :1], down - down[:, :1]
    up_rel_ep, _ = episode_means(up_rel, True)
    down_rel_ep, _ = episode_means(down_rel, False)
    p_nat_rel = np.zeros((50, 7)); d_nat_rel = np.zeros((50, 7))
    for h in range(50):
        for j in range(7):
            _, p_nat_rel[h, j] = stats.ttest_ind(up_rel_ep[:, h, j], down_rel_ep[:, h, j], equal_var=False)
            pooled = np.sqrt((up_rel_ep[:, h, j].var(ddof=1) + down_rel_ep[:, h, j].var(ddof=1)) / 2)
            d_nat_rel[h, j] = (up_rel_ep[:, h, j].mean() - down_rel_ep[:, h, j].mean()) / pooled if pooled > 0 else 0.0
    q_nat_rel = fdr_bh(p_nat_rel)

    # Counterfactual paired test: same images/states and same diffusion seed.
    p_cf = np.zeros((50, 7)); t_cf = np.zeros((50, 7))
    for h in range(50):
        for j in range(7):
            t, p = stats.ttest_1samp(cf_diff[:, h, j], 0.0)
            t_cf[h, j], p_cf[h, j] = t, p
    q_cf = fdr_bh(p_cf)
    cf_rel = (cf_up - cf_up[:, :1]) - (cf_down - cf_down[:, :1])

    stats_out = {
        "natural_episode_level": {
            "label_up_episodes": len(up_ep), "label_down_episodes": len(down_ep),
            "raw_p_lt_0.05_cells": int(np.sum(p_nat < 0.05)),
            "fdr_q_lt_0.05_cells": int(np.sum(q_nat < 0.05)), "total_cells": 350,
            "fdr_fraction": float(np.mean(q_nat < 0.05)),
            "max_abs_cohen_d": float(np.max(np.abs(d_nat))),
            "mean_abs_cohen_d": float(np.mean(np.abs(d_nat))),
            "per_joint_fdr_fraction": {j: float(np.mean(q_nat[:, i] < 0.05)) for i, j in enumerate(JOINTS)},
            "relative_to_first_frame": {
                "fdr_q_lt_0.05_cells": int(np.sum(q_nat_rel < 0.05)), "fdr_fraction": float(np.mean(q_nat_rel < 0.05)),
                "mean_abs_cohen_d": float(np.mean(np.abs(d_nat_rel))), "max_abs_cohen_d": float(np.max(np.abs(d_nat_rel))),
                "mean_abs_difference_deg": float(np.abs((up_rel.mean(axis=0) - down_rel.mean(axis=0))).mean()),
                "final_frame_mean_abs_difference_deg": float(np.abs(up_rel[:, -1].mean(axis=0) - down_rel[:, -1].mean(axis=0)).mean()),
            },
        },
        "counterfactual_paired": {
            "samples": int(len(cf_diff)), "raw_p_lt_0.05_cells": int(np.sum(p_cf < 0.05)),
            "fdr_q_lt_0.05_cells": int(np.sum(q_cf < 0.05)), "total_cells": 350,
            "fdr_fraction": float(np.mean(q_cf < 0.05)),
            "per_joint_fdr_fraction": {j: float(np.mean(q_cf[:, i] < 0.05)) for i, j in enumerate(JOINTS)},
            "mean_difference_deg_by_joint": cf_diff.mean(axis=(0, 1)).tolist(),
            "mean_abs_difference_deg_by_joint": np.abs(cf_diff).mean(axis=(0, 1)).tolist(),
            "p95_abs_difference_deg_by_joint": np.quantile(np.abs(cf_diff), 0.95, axis=(0, 1)).tolist(),
            "mean_abs_difference_deg_by_horizon": np.abs(cf_diff).mean(axis=(0, 2)).tolist(),
            "relative_to_first_frame": {
                "mean_abs_difference_deg": float(np.abs(cf_rel).mean()),
                "horizon_20_mean_abs_difference_deg": float(np.abs(cf_rel[:, :20]).mean()),
                "final_frame_mean_abs_difference_deg": float(np.abs(cf_rel[:, -1]).mean()),
                "mean_abs_difference_deg_by_joint": np.abs(cf_rel).mean(axis=(0, 1)).tolist(),
            },
            "action_blocks": {
                "left_arm_mean_abs_difference_deg_by_joint": np.rad2deg(np.abs(data["counterfactual_up"][:, :, :7] - data["counterfactual_down"][:, :, :7])).mean(axis=(0, 1)).tolist(),
                "right_arm_mean_abs_difference_deg_by_joint": np.abs(cf_diff).mean(axis=(0, 1)).tolist(),
                "gripper_mean_abs_difference_by_joint": np.abs(data["counterfactual_up"][:, :, 14:] - data["counterfactual_down"][:, :, 14:]).mean(axis=(0, 1)).tolist(),
                "left_arm_mean_abs_difference_deg": float(np.rad2deg(np.abs(data["counterfactual_up"][:, :, :7] - data["counterfactual_down"][:, :, :7])).mean()),
                "right_arm_mean_abs_difference_deg": float(np.abs(cf_diff).mean()),
                "gripper_mean_abs_difference": float(np.abs(data["counterfactual_up"][:, :, 14:] - data["counterfactual_down"][:, :, 14:]).mean()),
            },
        },
    }
    (ROOT / "stat_tests.json").write_text(json.dumps(stats_out, indent=2))

    # Figure 1: natural predicted absolute trajectories.
    x = np.arange(1, 51)
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:7]):
        for arr, color, label in [(up, "#1f77b4", "label_up"), (down, "#d95f02", "label_down")]:
            mean = arr[:, :, j].mean(axis=0); se = arr[:, :, j].std(axis=0, ddof=1) / np.sqrt(len(arr))
            ax.plot(x, mean, color=color, lw=1.8, label=label)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=.15, linewidth=0)
        ax.set_title(JOINTS[j]); ax.set_ylabel("joint target (deg)"); ax.grid(alpha=.25)
    axes[7].axis("off")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-2].set_xlabel("predicted horizon (frame)"); axes[-1].set_xlabel("predicted horizon (frame)")
    fig.suptitle("Natural label-up vs label-down action trajectories (absolute right-arm targets)")
    fig.tight_layout()
    fig.savefig(ROOT / "natural_label_trajectory_comparison.png", dpi=160)
    plt.close(fig)

    # Same comparison after subtracting each sample's first predicted frame.
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:7]):
        for arr, color, label in [(up_rel, "#1f77b4", "label_up"), (down_rel, "#d95f02", "label_down")]:
            mean = arr[:, :, j].mean(axis=0); se = arr[:, :, j].std(axis=0, ddof=1) / np.sqrt(len(arr))
            ax.plot(x, mean, color=color, lw=1.8, label=label)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=.15, linewidth=0)
        ax.axhline(0, color="0.4", lw=.7); ax.set_title(JOINTS[j]); ax.set_ylabel("delta from frame 1 (deg)"); ax.grid(alpha=.25)
    axes[7].axis("off"); axes[0].legend(frameon=False, ncol=2, loc="best")
    axes[-2].set_xlabel("predicted horizon (frame)"); axes[-1].set_xlabel("predicted horizon (frame)")
    fig.suptitle("Relative natural trajectories (initial posture removed)"); fig.tight_layout()
    fig.savefig(ROOT / "natural_relative_trajectory_comparison.png", dpi=160); plt.close(fig)

    # Figure 2: paired counterfactual effect (up prompt minus down prompt).
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:7]):
        mean = cf_diff[:, :, j].mean(axis=0); se = cf_diff[:, :, j].std(axis=0, ddof=1) / np.sqrt(len(cf_diff))
        ax.axhline(0, color="0.35", lw=0.8); ax.plot(x, mean, color="#6a3d9a", lw=1.8)
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color="#6a3d9a", alpha=.18, linewidth=0)
        ax.set_title(JOINTS[j]); ax.set_ylabel("up − down (deg)"); ax.grid(alpha=.25)
    axes[7].axis("off")
    axes[-2].set_xlabel("predicted horizon (frame)"); axes[-1].set_xlabel("predicted horizon (frame)")
    fig.suptitle("Paired counterfactual language effect on identical observations")
    fig.tight_layout()
    fig.savefig(ROOT / "counterfactual_prompt_effect.png", dpi=160)
    plt.close(fig)

    # Figure 3: compact effect-size summary.
    cf_abs = np.abs(cf_diff).mean(axis=(0, 1))
    p95 = np.quantile(np.abs(cf_diff), .95, axis=(0, 1))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos = np.arange(7); ax.bar(pos, cf_abs, color="#6a3d9a", alpha=.85, label="mean |up−down|")
    ax.scatter(pos, p95, color="#e31a1c", marker="o", label="P95 |up−down|")
    ax.axhline(1, color="0.4", ls="--", lw=1, label="1° reference")
    ax.set_xticks(pos, JOINTS); ax.set_ylabel("difference (deg)"); ax.set_title("Paired prompt effect by right-arm joint")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=3); fig.tight_layout()
    fig.savefig(ROOT / "counterfactual_effect_by_joint.png", dpi=160)
    plt.close(fig)
    print(json.dumps(stats_out, indent=2))


if __name__ == "__main__":
    main()
