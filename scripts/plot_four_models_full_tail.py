"""Plot model and recorded actions from first closure through episode end."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(
    "/home/bjtc/Sophix/lingbot-vla/output/separated_model/"
    "postclose_full_tail_eval_20260930"
)
COLORS = {True: "#1f77b4", False: "#d95f02"}


def plot_model(name: str):
    out = ROOT / name
    with np.load(out / "paired_trajectories.npz") as saved:
        prediction = np.rad2deg(saved["prediction"][:, :, 7:14])
        truth = np.rad2deg(saved["truth"][:, :, 7:14])
        anchor = np.rad2deg(saved["anchor"][:, 7:14])[:, None, :]
        labels = saved["label_up"].astype(bool)
        lengths = saved["lengths"]
    manifest = json.loads((out / "manifest.json").read_text())
    chunk = int(manifest["chunk_size"])
    frames = prediction.shape[1]
    x = np.arange(frames)
    figure, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for joint, axis in enumerate(axes.ravel()[:7]):
        for label in (True, False):
            mask = labels == label
            if not mask.any():
                continue
            label_text = "up" if label else "down"
            color = COLORS[label]
            axis.plot(
                x,
                np.nanmean((prediction - anchor)[mask, :, joint], axis=0),
                color=color,
                linewidth=1.5,
                label=f"{label_text} model",
            )
            axis.plot(
                x,
                np.nanmean((truth - anchor)[mask, :, joint], axis=0),
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"{label_text} recorded",
            )
        for boundary in np.arange(chunk, frames, chunk):
            axis.axvline(boundary, color=".7", linewidth=0.55, linestyle=":")
        axis.set_title(f"Right joint {joint + 1}")
        axis.set_ylabel("Change from first closure (deg)")
        axis.set_xlabel("Frame from first closure to episode end")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9, ncol=2, frameon=False)
    count_axis = axes.ravel()[-1]
    count_axis.plot(x, (lengths[:, None] > x).sum(axis=0), color="#555555")
    count_axis.set_title("Episodes remaining")
    count_axis.set_ylabel("Episode count")
    count_axis.set_xlabel("Frame from first closure to episode end")
    count_axis.grid(alpha=0.2)
    figure.suptitle(
        f"{name}: first closure to episode end; {len(labels)} held-out episodes\n"
        f"tail={lengths.min()}-{lengths.max()} frames; native chunk={chunk}; "
        "recorded observations"
    )
    figure.tight_layout()
    figure.savefig(out / "model_vs_recorded_full_tail.png", dpi=160)
    plt.close(figure)


def main():
    summary = json.loads((ROOT / "metrics_summary.json").read_text())
    for name in summary:
        plot_model(name)


if __name__ == "__main__":
    main()
