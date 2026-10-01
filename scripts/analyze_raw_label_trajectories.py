"""Analyze original marked_data_0916 action trajectories by label orientation.

This reads the raw LeRobot parquet actions, not model predictions.  Episodes are
weighted equally; each episode is linearly resampled to a common normalized
progress axis so long episodes do not dominate the comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy import stats
from matplotlib.lines import Line2D


DATA = Path("/home/bjtc/Sophix/datasets/marked_data_0916")
OUT = Path("/home/bjtc/Sophix/lingbot-vla/output/marked_full0919/raw_label_trajectory_analysis_20260921")
ROLLOUT_ROOT = Path("/home/bjtc/Sophix/g1/runtime_rollouts")
N_POINTS = 100
POSTCLOSE_POINTS = 200
JOINTS = [f"right_j{i}" for i in range(1, 8)]
LABEL_COLORS = {"label_up": "#1f77b4", "label_down": "#d95f02"}


def fdr_bh(p):
    p = np.asarray(p, dtype=float)
    flat = np.where(np.isfinite(p), p, 1.0).ravel()
    order = np.argsort(flat); ranked = flat[order]
    q = np.empty_like(ranked); running = 1.0
    for i in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[i] * len(ranked) / (i + 1)); q[i] = running
    out = np.empty_like(flat); out[order] = q
    return out.reshape(p.shape)


def resample_episode(action, n=N_POINTS):
    action = np.asarray(action, dtype=np.float64)
    t_old = np.linspace(0.0, 1.0, len(action))
    t_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, action.shape[1]), dtype=np.float64)
    for j in range(action.shape[1]):
        # Arm targets are continuous.  Gripper values are also interpolated
        # only for a smooth visualization; gripper conclusions use endpoints.
        out[:, j] = np.interp(t_new, t_old, action[:, j])
    return out


def load_all():
    labels = {}
    with (DATA / "meta/annotations/episode_labels.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            labels[int(r["episode_index"])] = r["extra_labels"]["self_annotation"]
    episode_rows = []
    for p in sorted((DATA / "meta/episodes").glob("**/*.parquet")):
        episode_rows.extend(pq.read_table(p).to_pylist())
    episodes = {}
    checks = {"episodes": 0, "frames": 0, "bad_length": [], "bad_index": [], "nonfinite": []}
    for row in sorted(episode_rows, key=lambda r: int(r["episode_index"])):
        ep = int(row["episode_index"]); n = int(row["length"])
        p = DATA / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
        tab = pq.read_table(p).to_pydict()
        a = np.asarray(tab["action"], dtype=np.float64)
        eps = np.asarray(tab["episode_index"], dtype=np.int64)
        frames = np.asarray(tab["frame_index"], dtype=np.int64)
        if len(a) != n: checks["bad_length"].append(ep)
        ids = np.asarray(tab["index"], dtype=np.int64)
        if (not np.array_equal(frames, np.arange(len(frames))) or not np.all(eps == ep)
                or not np.array_equal(ids, np.arange(row["dataset_from_index"], row["dataset_to_index"]))):
            checks["bad_index"].append(ep)
        if not np.isfinite(a).all(): checks["nonfinite"].append(ep)
        # A gripper-command phase proxy, not verified object contact or success.
        closed = a[:, 15] > .5
        starts = np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1
        ends = np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1
        phase = None
        if (not closed[0] and not closed[-1] and len(starts) == 1
                and len(ends) == 1 and ends[0] - starts[0] >= 15):
            phase = (int(starts[0]), int(ends[0]))
        postclose = None
        if phase and phase[0] + POSTCLOSE_POINTS <= len(a):
            # Keep the native 30-Hz samples.  The reference rollout figure
            # starts at the first closure frame and shows the next 200 frames.
            postclose = a[phase[0]:phase[0] + POSTCLOSE_POINTS]
        episodes[ep] = {"action": resample_episode(a), "length": len(a), "label_up": bool(labels[ep]["grasp_label_up"]),
                        "regrasp": bool(labels[ep]["had_regrasp"]),
                        "command_phase": phase,
                        "phase_action": resample_episode(a[phase[0]:phase[1]+1]) if phase else None,
                        "postclose_action": postclose,
                        "right_range_deg": np.rad2deg(np.ptp(a[:, 7:14], axis=0)).tolist(),
                        "right_travel_deg": np.rad2deg(np.abs(np.diff(a[:, 7:14], axis=0)).sum(axis=0)).tolist()}
        checks["episodes"] += 1; checks["frames"] += len(a)
    assert checks["episodes"] == 1064 and checks["frames"] == 748377
    assert not any(checks[k] for k in ("bad_length", "bad_index", "nonfinite")), checks
    return episodes, checks


def effect_stats(up, down):
    # Arrays: episodes x normalized time x 7 right-arm joints, in degrees.
    p = np.empty((N_POINTS, 7)); d = np.empty_like(p)
    for t in range(N_POINTS):
        for j in range(7):
            if np.ptp(up[:, t, j]) == 0 and np.ptp(down[:, t, j]) == 0:
                p[t, j] = 1.0 if up[0, t, j] == down[0, t, j] else 0.0
            else:
                _, p[t, j] = stats.ttest_ind(up[:, t, j], down[:, t, j], equal_var=False)
            pooled = np.sqrt((up[:, t, j].var(ddof=1) + down[:, t, j].var(ddof=1)) / 2)
            d[t, j] = (up[:, t, j].mean() - down[:, t, j].mean()) / pooled if pooled > 0 else 0.0
    q = fdr_bh(p)
    mean_diff = up.mean(axis=0) - down.mean(axis=0)
    return {
        "raw_p_lt_0.05_cells": int(np.sum(p < .05)),
        "fdr_q_lt_0.05_cells": int(np.sum(q < .05)),
        "total_cells": int(N_POINTS * 7),
        "fdr_fraction": float(np.mean(q < .05)),
        "per_joint_fdr_fraction": {JOINTS[j]: float(np.mean(q[:, j] < .05)) for j in range(7)},
        "mean_abs_cohen_d": float(np.mean(np.abs(d))),
        "max_abs_cohen_d": float(np.max(np.abs(d))),
        "mean_difference_deg_by_joint": mean_diff.mean(axis=0).tolist(),
        "mean_abs_difference_deg": float(np.abs(mean_diff).mean()),
        "mean_abs_difference_deg_by_joint": np.abs(mean_diff).mean(axis=0).tolist(),
        "max_abs_group_mean_difference_deg_by_joint": np.max(np.abs(mean_diff), axis=0).tolist(),
        "peak_progress_percent_by_joint": (np.argmax(np.abs(mean_diff), axis=0) * 100 / (N_POINTS-1)).tolist(),
        "final_frame_mean_difference_deg_by_joint": mean_diff[-1].tolist(),
        "final_frame_mean_abs_difference_deg": float(np.abs(mean_diff[-1]).mean()),
        "p95_abs_difference_deg_by_joint_over_time": np.quantile(np.abs(mean_diff), .95, axis=0).tolist(),
        "p_values": p.tolist(), "q_values": q.tolist(),
    }


def summarize_group(actions):
    # action arrays are episodes x time x 16, radians for arms.
    right = np.rad2deg(actions[:, :, 7:14])
    rel = right - right[:, :1]
    return {
        "episodes": int(len(actions)),
        "right_arm_absolute_mean_deg": right.mean(axis=0).tolist(),
        "right_arm_absolute_std_deg": right.std(axis=0).tolist(),
        "right_arm_relative_mean_deg": rel.mean(axis=0).tolist(),
        "right_arm_relative_final_mean_deg": rel[:, -1].mean(axis=0).tolist(),
        "right_arm_relative_final_mean_abs_deg": float(np.abs(rel[:, -1]).mean()),
        "right_arm_path_length_mean_deg": float(np.abs(np.diff(right, axis=1)).sum(axis=(1, 2)).mean()),
        "right_arm_rms_step_mean_deg": float(np.sqrt(np.square(np.diff(right, axis=1)).mean(axis=(1, 2))).mean()),
        "left_arm_relative_final_mean_abs_deg": float(np.rad2deg(np.abs(actions[:, -1, :7] - actions[:, 0, :7])).mean()),
        "gripper_initial_mean": actions[:, 0, 14:].mean(axis=0).tolist(),
        "gripper_final_mean": actions[:, -1, 14:].mean(axis=0).tolist(),
        "gripper_final_minus_initial_mean": (actions[:, -1, 14:] - actions[:, 0, 14:]).mean(axis=0).tolist(),
    }


def plot_relative(result_arrays):
    x = np.linspace(0, 100, N_POINTS)
    fig, axes = plt.subplots(2, 7, figsize=(20, 6.5), sharex=True)
    for row, subset in enumerate(("all", "non_regrasp")):
        up, down = result_arrays[subset]
        up_r = np.rad2deg(up[:, :, 7:14]); down_r = np.rad2deg(down[:, :, 7:14])
        up_r -= up_r[:, :1]; down_r -= down_r[:, :1]
        for j, ax in enumerate(axes[row]):
            for arr, color, label in ((up_r, "#1f77b4", "label_up"), (down_r, "#d95f02", "label_down")):
                mean = arr[:, :, j].mean(axis=0); se = arr[:, :, j].std(axis=0, ddof=1) / np.sqrt(len(arr))
                ax.plot(x, mean, color=color, lw=1.5, label=label)
                ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=.14, linewidth=0)
            ax.axhline(0, color="0.4", lw=.6); ax.grid(alpha=.22); ax.set_title(JOINTS[j])
            if j == 0: ax.set_ylabel(f"{subset}\nΔ from start (deg)")
            if row == 1: ax.set_xlabel("episode progress (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Original training action trajectories: label-up vs label-down")
    fig.tight_layout(); fig.savefig(OUT / "raw_relative_trajectories.png", dpi=160); plt.close(fig)


def load_real_rollout_overlays():
    """Load the ten rollout curves used by plot_real_rollouts_postclose.py."""
    overlays = []
    sources = [(str(i), "label_down") for i in range(1, 6)]
    sources += [(f"prompt_changed_{i}", "label_up") for i in range(1, 6)]
    for name, label_name in sources:
        table = pq.read_table(ROLLOUT_ROOT / name / "aligned/steps.parquet").to_pydict()
        action = np.asarray(table["action"], dtype=np.float64)
        measured = np.asarray(table["measured_next_state"], dtype=np.float64)
        gripper = action[:, 15]
        crossings = np.flatnonzero((gripper[1:] > .5) & (gripper[:-1] <= .5)) + 1
        if len(crossings) == 0:
            raise RuntimeError(f"rollout {name}: no right-gripper closure crossing")
        close = int(crossings[0])
        if close + POSTCLOSE_POINTS > len(action):
            raise RuntimeError(f"rollout {name}: less than 200 frames after closure")
        overlays.append({
            "name": name, "label_name": label_name, "closure_frame": close,
            "command_deg": np.rad2deg(action[close:close + POSTCLOSE_POINTS, 7:14]),
            "measured_deg": np.rad2deg(measured[close:close + POSTCLOSE_POINTS, 7:14]),
        })
    return overlays


def plot_postclose_200(postclose_arrays, rollout_overlays):
    """Plot absolute source joint angles on the rollout figure's 200-frame axis."""
    x = np.arange(1, POSTCLOSE_POINTS + 1)
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True)
    axes = axes.ravel()
    for j, ax in enumerate(axes[:7]):
        for arr, label_name in zip(postclose_arrays, ("label_up", "label_down")):
            color = LABEL_COLORS[label_name]
            label = label_name.replace("_", " ")
            values = np.rad2deg(arr[:, :, 7 + j])
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            for trajectory in values:
                ax.plot(x, trajectory, color=color, alpha=.075, lw=.65,
                        zorder=1, rasterized=True)
            ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                            color=color, alpha=.15, linewidth=0, zorder=2)
            ax.plot(x, mean, color=color, lw=2.0, zorder=3,
                    label=f"{label} (n={len(values)})")
        # Highlight the ten real rollout curves from the reference figure.
        # Color now follows label semantics rather than prompt-version semantics.
        for rec in rollout_overlays:
            color = LABEL_COLORS[rec["label_name"]]
            ax.plot(x, rec["command_deg"][:, j], color=color, lw=1.55,
                    alpha=.82, zorder=4)
            ax.plot(x, rec["measured_deg"][:, j], color=color, lw=1.25,
                    ls="--", alpha=.72, zorder=4)
        for boundary in (50.5, 100.5, 150.5):
            ax.axvline(boundary, color="0.65", lw=.7, ls=":", zorder=0)
        ax.set_title(f"Right joint {j + 1}")
        ax.set_ylabel("Joint angle (deg)")
        ax.grid(alpha=.2)
    axes[7].axis("off")
    axes[6].set_xlabel("Frame after first right-gripper closure (1-200)")
    source_handles, source_labels = axes[0].get_legend_handles_labels()
    semantic_handles = [
        Line2D([0], [0], color="0.2", lw=1.55, label="highlighted rollout command"),
        Line2D([0], [0], color="0.2", lw=1.25, ls="--", label="highlighted rollout measured next state"),
    ]
    axes[0].legend(source_handles + semantic_handles, source_labels + [h.get_label() for h in semantic_handles],
                   frameon=False, fontsize=8, loc="best")
    fig.suptitle(
        "Raw source action trajectories | single_grip_non_regrasp | post-closure 200 frames\n"
        "Blue: label up | Orange: label down | bold overlays: real rollouts 1–5 per label",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(OUT / "single_grip_non_regrasp_postclose_200.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    episodes, checks = load_all()
    arrays = {}
    for subset, predicate in (("all", lambda e: True), ("non_regrasp", lambda e: not e["regrasp"])):
        up = np.stack([e["action"] for e in episodes.values() if predicate(e) and e["label_up"]])
        down = np.stack([e["action"] for e in episodes.values() if predicate(e) and not e["label_up"]])
        arrays[subset] = (up, down)
    phase_up = [e for e in episodes.values() if not e["regrasp"] and e["label_up"] and e["command_phase"]]
    phase_down = [e for e in episodes.values() if not e["regrasp"] and not e["label_up"] and e["command_phase"]]
    arrays["single_grip_non_regrasp"] = (np.stack([e["phase_action"] for e in phase_up]), np.stack([e["phase_action"] for e in phase_down]))
    postclose_arrays = (
        np.stack([e["postclose_action"] for e in phase_up]),
        np.stack([e["postclose_action"] for e in phase_down]),
    )
    rollout_overlays = load_real_rollout_overlays()
    result = {
        "dataset": str(DATA), "fps": 30, "resampled_points": N_POINTS,
        "protocol": "Raw action targets from every episode; equal episode weighting; linear resampling to normalized episode progress; arm values converted from radians to degrees. Relative trajectories subtract each episode's first action frame.",
        "integrity": checks,
        "phase_proxy": "Non-regrasp episodes with exactly one bounded right-gripper action >0.5 interval lasting at least 15 frames; align close-command crossing to open-command crossing. This is not verified physical grasp or placement.",
        "groups": {},
    }
    for subset, (up, down) in arrays.items():
        result["groups"][subset] = {
            "label_up": summarize_group(up), "label_down": summarize_group(down),
            "comparison_absolute": effect_stats(np.rad2deg(up[:, :, 7:14]), np.rad2deg(down[:, :, 7:14])),
            "comparison_relative": effect_stats(
                np.rad2deg(up[:, :, 7:14] - up[:, :1, 7:14]),
                np.rad2deg(down[:, :, 7:14] - down[:, :1, 7:14])),
        }
    # Remove dense p/q surfaces from the headline JSON; retain them separately.
    dense = {}
    for subset in arrays:
        for mode in ("comparison_absolute", "comparison_relative"):
            dense[f"{subset}_{mode}"] = {
                "p_values": result["groups"][subset][mode].pop("p_values"),
                "q_values": result["groups"][subset][mode].pop("q_values"),
            }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2))
    (OUT / "significance_surfaces.json").write_text(json.dumps(dense, indent=2))
    np.savez_compressed(OUT / "resampled_raw_actions.npz", **{
        f"{subset}_label_up": up for subset, (up, down) in arrays.items()
    }, **{f"{subset}_label_down": down for subset, (up, down) in arrays.items()})
    np.savez_compressed(
        OUT / "postclose_200_raw_actions.npz",
        label_up=postclose_arrays[0], label_down=postclose_arrays[1],
    )
    # Preserve episode identities, lengths and unresampled range/travel for audit.
    (OUT / "episode_metrics.json").write_text(json.dumps([
        {"episode": ep, **{k: v for k, v in e.items() if k not in ("action", "phase_action", "postclose_action")}}
        for ep, e in episodes.items()], indent=2))
    robust = {}
    for subset in ("all", "non_regrasp"):
        groups = [[e for e in episodes.values() if (subset == "all" or not e["regrasp"]) and e["label_up"] == label] for label in (True, False)]
        robust[subset] = {}
        for feature in ("right_range_deg", "right_travel_deg", "length"):
            a, b = (np.asarray([e[feature] for e in g], float) for g in groups)
            _, p = stats.ttest_ind(a, b, axis=0, equal_var=False)
            robust[subset][feature] = {"up_mean": a.mean(0).tolist(), "down_mean": b.mean(0).tolist(), "q": fdr_bh(p).tolist()}
    (OUT / "timing_independent_metrics.json").write_text(json.dumps(robust, indent=2))
    plot_relative(arrays)
    plot_postclose_200(postclose_arrays, rollout_overlays)
    for subset in ("non_regrasp", "single_grip_non_regrasp"):
        # Match the readable small-multiple layout used for the real-rollout
        # figure, while retaining every normalized source episode.
        fig, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True)
        x = np.linspace(0, 100, N_POINTS)
        for j, ax in enumerate(axes.ravel()[:7]):
            for arr, label_name in zip(arrays[subset], ("label_up", "label_down")):
                color = LABEL_COLORS[label_name]
                label = label_name.replace("_", " ")
                a = np.rad2deg(arr[:, :, j+7] - arr[:, :1, j+7])
                mean, se = a.mean(0), a.std(0, ddof=1)/np.sqrt(len(a))
                # Show every normalized source episode as a faint hairline.  The
                # summary statistics remain on top so the distribution is visible
                # without losing the mean/CI shown in the original figure.
                for trajectory in a:
                    ax.plot(x, trajectory, color=color, alpha=.075, lw=.65,
                            zorder=1, rasterized=True)
                ax.fill_between(x, mean-1.96*se, mean+1.96*se, color=color,
                                alpha=.15, linewidth=0, zorder=2)
                ax.plot(x, mean, color=color, lw=2.0, zorder=3,
                        label=f"{label} (n={len(a)})")
            ax.set_title(JOINTS[j]); ax.set_ylabel("Change from phase start (deg)")
            ax.set_xlabel("Episode progress (%)" if subset == "non_regrasp" else "Close-to-open command progress (%)")
            ax.grid(alpha=.2)
        axes.ravel()[-1].axis("off"); axes[0,0].legend(frameon=False)
        fig.suptitle("Raw action trajectories | " + subset + " | all source episodes + shaded 95% CI of mean", fontsize=14)
        fig.tight_layout(); fig.savefig(OUT / f"{subset}_relative.png", dpi=170); plt.close(fig)
    print(json.dumps({"integrity": checks, "groups": {
        s: {"label_up": result["groups"][s]["label_up"]["episodes"],
            "label_down": result["groups"][s]["label_down"]["episodes"],
            "relative": result["groups"][s]["comparison_relative"]}
        for s in arrays
    }}, indent=2))


if __name__ == "__main__":
    main()
