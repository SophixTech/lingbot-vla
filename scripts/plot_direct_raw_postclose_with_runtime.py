#!/usr/bin/env python3
"""Direct-raw action post-closure plot with 20 physical-rollout overlays.

Raw actions are rebuilt from ``data/package_raw_data`` using the same public
alignment contract as the exporter: left-wrist JPEG timestamps are the master
timeline, joint actions are linearly interpolated, and gripper commands use
zero-order hold.  No converted Parquet actions are used for the source curves.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path("/home/bjtc/Sophix")
RAW_ROOT = ROOT / "data/package_raw_data"
LABEL_FILE = ROOT / "datasets/marked_data_0916/meta/annotations/episode_labels.jsonl"
EXCLUSIONS_FILE = ROOT / "cleanup_reports/bad9_20260923_143636/exclusions.json"
ROLLOUT_ROOT = ROOT / "g1/runtime_rollouts"
OUT = ROOT / "lingbot-vla/output/marked_full0919/raw_label_trajectory_analysis_20261001_direct_raw"
POSTCLOSE_POINTS = 200
LABEL_COLORS = {"label_up": "#1f77b4", "label_down": "#d95f02"}


def jpeg_timestamps(directory: Path) -> np.ndarray:
    entries = []
    for path in directory.glob("*.jpg"):
        try:
            entries.append((int(path.stem), path))
        except ValueError:
            continue
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise ValueError(f"empty left-wrist image stream: {directory}")
    return np.asarray([timestamp for timestamp, _ in entries], dtype=np.int64)


def interpolate(values: np.ndarray, timestamps: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(timestamps) or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("invalid stream for linear interpolation")
    origin = int(target[0])
    x = (timestamps - origin).astype(np.float64)
    t = (target - origin).astype(np.float64)
    return np.column_stack([np.interp(t, x, values[:, column]) for column in range(values.shape[1])])


def zero_order_hold(values: np.ndarray, timestamps: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(timestamps) or not len(timestamps) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("invalid stream for zero-order hold")
    index = np.searchsorted(timestamps, target, side="right") - 1
    return values[np.clip(index, 0, len(values) - 1)]


def direct_raw_action(episode_id: str) -> np.ndarray:
    root = RAW_ROOT / episode_id
    target = jpeg_timestamps(root / "camera/hand_left/color")
    h5_path = root / "record/raw_joints.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"{episode_id}: missing {h5_path.name}")
    with h5py.File(h5_path, "r") as h5:
        def read(name: str) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.asarray(h5[f"{name}/timestamp"][:], dtype=np.int64),
                np.asarray(h5[f"{name}/position"][:], dtype=np.float64),
            )

        joint_t, joint = read("action/joint")
        left_t, left = read("action/left_effector")
        right_t, right = read("action/right_effector")
    action = np.concatenate([
        interpolate(joint, joint_t, target),
        zero_order_hold(left, left_t, target),
        zero_order_hold(right, right_t, target),
    ], axis=1)
    if action.shape[1] != 16 or not np.isfinite(action).all():
        raise ValueError(f"{episode_id}: expected finite 16-D action, got {action.shape}")
    return action


def labels_and_exclusions() -> tuple[dict[str, dict], list[dict]]:
    exclusions = json.loads(EXCLUSIONS_FILE.read_text())
    excluded_ids = {item["source_episode_id"] for item in exclusions}
    labels = {}
    for line in LABEL_FILE.read_text().splitlines():
        row = json.loads(line)
        episode_id = row["source_episode_id"]
        if episode_id in excluded_ids:
            raise ValueError(f"excluded episode unexpectedly appears in annotation labels: {episode_id}")
        label = row["extra_labels"]["self_annotation"]
        labels[episode_id] = {
            "source_episode_index": int(row["source_episode_index"]),
            "label_up": bool(label["grasp_label_up"]),
            "had_regrasp": bool(label["had_regrasp"]),
        }
    if len(labels) != 1055:
        raise ValueError(f"expected 1055 approved labeled source episodes, got {len(labels)}")
    still_present = [item["source_episode_id"] for item in exclusions if (RAW_ROOT / item["source_episode_id"]).exists()]
    if still_present:
        raise ValueError(f"excluded raw source directories still present: {still_present}")
    return labels, exclusions


def first_bounded_close(action: np.ndarray) -> int | None:
    closed = action[:, 15] > 0.5
    starts = np.flatnonzero(np.diff(closed.astype(np.int8)) == 1) + 1
    ends = np.flatnonzero(np.diff(closed.astype(np.int8)) == -1) + 1
    if closed[0] or closed[-1] or len(starts) != 1 or len(ends) != 1 or ends[0] - starts[0] < 15:
        return None
    return int(starts[0])


def load_source_postclose(labels: dict[str, dict]) -> tuple[dict[str, list[np.ndarray]], dict]:
    groups = {"label_up": [], "label_down": []}
    audit = {"labeled_raw_present": 0, "regrasp_excluded": [], "invalid_closure_phase": [], "short_postclose": [], "included": []}
    for episode_id, label in sorted(labels.items(), key=lambda item: item[1]["source_episode_index"]):
        if not (RAW_ROOT / episode_id).is_dir():
            raise FileNotFoundError(f"approved labeled raw source missing: {episode_id}")
        audit["labeled_raw_present"] += 1
        if label["had_regrasp"]:
            audit["regrasp_excluded"].append(label["source_episode_index"])
            continue
        action = direct_raw_action(episode_id)
        close = first_bounded_close(action)
        if close is None:
            audit["invalid_closure_phase"].append(label["source_episode_index"])
            continue
        if close + POSTCLOSE_POINTS > len(action):
            audit["short_postclose"].append(label["source_episode_index"])
            continue
        key = "label_up" if label["label_up"] else "label_down"
        groups[key].append(action[close : close + POSTCLOSE_POINTS])
        audit["included"].append({
            "source_episode_index": label["source_episode_index"],
            "source_episode_id": episode_id,
            "label": key,
            "closure_frame": close,
        })
    if not groups["label_up"] or not groups["label_down"]:
        raise ValueError("empty post-closure source group")
    return {key: np.stack(value) for key, value in groups.items()}, audit


def load_runtime_overlays() -> tuple[list[dict], dict]:
    overlays = []
    details = {}
    for number in range(1, 21):
        name = f"0930-{number}"
        label = "label_down" if number <= 10 else "label_up"
        rows = [json.loads(line) for line in (ROLLOUT_ROOT / name / "steps.jsonl").read_text().splitlines() if line]
        command = np.asarray([row["published_action"] for row in rows], dtype=np.float64)
        measured = np.asarray([row["measured_arm"] for row in rows], dtype=np.float64)
        crossings = np.flatnonzero(np.diff((command[:, 15] > 0.5).astype(np.int8)) == 1) + 1
        if not len(crossings):
            raise ValueError(f"{name}: no right-gripper closing edge")
        close = int(crossings[0])
        if close + POSTCLOSE_POINTS > len(rows):
            raise ValueError(f"{name}: fewer than {POSTCLOSE_POINTS} samples after closure")
        overlays.append({
            "name": name,
            "label": label,
            "command_deg": np.rad2deg(command[close : close + POSTCLOSE_POINTS, 7:14]),
            "measured_deg": np.rad2deg(measured[close : close + POSTCLOSE_POINTS, 7:14]),
        })
        details[name] = {"label": label, "closure_sample_zero_based": close, "rows": len(rows)}
    return overlays, details


def plot(source: dict[str, np.ndarray], overlays: list[dict]) -> None:
    x = np.arange(1, POSTCLOSE_POINTS + 1)
    figure, axes = plt.subplots(4, 2, figsize=(15, 17), sharex=True)
    axes = axes.ravel()
    for joint, axis in enumerate(axes[:7]):
        for label in ("label_up", "label_down"):
            values = np.rad2deg(source[label][:, :, 7 + joint])
            mean = values.mean(axis=0)
            se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
            color = LABEL_COLORS[label]
            for trajectory in values:
                axis.plot(x, trajectory, color=color, alpha=0.075, lw=0.65, zorder=1, rasterized=True)
            axis.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.15, linewidth=0, zorder=2)
            axis.plot(x, mean, color=color, lw=2.0, zorder=3, label=f"{label.replace('_', ' ')} (n={len(values)})")
        for rollout in overlays:
            color = LABEL_COLORS[rollout["label"]]
            axis.plot(x, rollout["command_deg"][:, joint], color=color, lw=1.55, alpha=0.82, zorder=4)
            axis.plot(x, rollout["measured_deg"][:, joint], color=color, lw=1.25, ls="--", alpha=0.72, zorder=4)
        for boundary in (50.5, 100.5, 150.5):
            axis.axvline(boundary, color="0.65", lw=0.7, ls=":", zorder=0)
        axis.set_title(f"Right joint {joint + 1}")
        axis.set_ylabel("Joint angle (deg)")
        axis.grid(alpha=0.2)
    axes[7].axis("off")
    axes[6].set_xlabel("Frame after first right-gripper closure (1-200)")
    source_handles, source_labels = axes[0].get_legend_handles_labels()
    semantic_handles = [
        Line2D([0], [0], color="0.2", lw=1.55, label="current runtime rollout command"),
        Line2D([0], [0], color="0.2", lw=1.25, ls="--", label="current runtime rollout measured arm"),
    ]
    axes[0].legend(source_handles + semantic_handles, source_labels + [handle.get_label() for handle in semantic_handles],
                   frameon=False, fontsize=8, loc="best")
    figure.suptitle(
        "Direct raw-source action trajectories | single_grip_non_regrasp | post-closure 200 frames\n"
        "Blue: label up | Orange: label down | bold overlays: current runtime rollouts 10 per label",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(OUT / "direct_raw_single_grip_non_regrasp_postclose_200_with_runtime20.png", dpi=170)
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels, exclusions = labels_and_exclusions()
    raw_ids = {path.name for path in RAW_ROOT.iterdir() if path.is_dir()}
    source, source_audit = load_source_postclose(labels)
    overlays, overlay_audit = load_runtime_overlays()
    plot(source, overlays)
    unlabelled_raw = sorted(raw_ids - set(labels))
    result = {
        "raw_root": str(RAW_ROOT),
        "source_action": "rebuilt directly from raw_joints.h5 on camera/hand_left/color JPEG timestamps; joint=linear interpolation; gripper=zero-order hold",
        "excluded_source_episodes": exclusions,
        "raw_directory_count": len(raw_ids),
        "raw_directories_without_approved_label": unlabelled_raw,
        "selection": source_audit,
        "source_group_counts": {label: len(values) for label, values in source.items()},
        "runtime_overlays": overlay_audit,
        "output_image": str(OUT / "direct_raw_single_grip_non_regrasp_postclose_200_with_runtime20.png"),
    }
    (OUT / "direct_raw_single_grip_non_regrasp_postclose_200_with_runtime20.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(OUT / "direct_raw_postclose_200_actions.npz", label_up=source["label_up"], label_down=source["label_down"])
    print(json.dumps({"source_group_counts": result["source_group_counts"], "output_image": result["output_image"], "unlabelled_raw": unlabelled_raw}, indent=2))


if __name__ == "__main__":
    main()
