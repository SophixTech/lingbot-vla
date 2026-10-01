"""Plot four-model post-closure trajectories and metrics."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output/separated_model/postclose_eval_20260930")
COLORS = {True: "#1f77b4", False: "#d95f02"}

def plot_model(name, metrics):
    out = ROOT / name; z = np.load(out / "paired_trajectories.npz"); m = json.loads((out / "manifest.json").read_text())
    pred = np.rad2deg(z["prediction"][:, :, 7:14])
    truth = np.rad2deg(z["truth"][:, :, 7:14])
    anchor = np.rad2deg(z["anchor"][:, 7:14])[:, None, :]
    labels = z["label_up"].astype(bool); chunk = int(m["chunk_size"]); frames = pred.shape[1]; x = np.arange(1, frames + 1)
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for j, ax in enumerate(axes.ravel()[:7]):
        for label in (True, False):
            mask = labels == label
            if not mask.any(): continue
            text = "up" if label else "down"; color = COLORS[label]
            ax.plot(x, (pred-anchor)[mask, :, j].mean(0), color=color, lw=1.5, label=f"{text} model")
            ax.plot(x, (truth-anchor)[mask, :, j].mean(0), color=color, ls="--", lw=1.5, label=f"{text} recorded")
        for b in np.arange(chunk + .5, frames, chunk): ax.axvline(b, color=".6", lw=.7, ls=":")
        ax.set_title(f"Right joint {j+1}"); ax.set_ylabel("Change from closure (deg)"); ax.set_xlabel("Frame after closure (1-200)"); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9, ncol=2, frameon=False); axes.ravel()[-1].axis("off")
    fig.suptitle(f"{name}: held-out {len(labels)} non-regrasp episodes (up={labels.sum()}, down={(~labels).sum()})\n{frames//chunk} predictions; native chunk={chunk}; recorded observations")
    fig.tight_layout(); fig.savefig(out / "model_vs_recorded_trajectories.png", dpi=160); plt.close(fig)

def main():
    summary = json.loads((ROOT / "metrics_summary.json").read_text())
    for name, metrics in summary.items(): plot_model(name, metrics)
    names = list(summary); labels = ["label-up", "label-down", "mixed-50", "mixed-25"]; x = np.arange(len(names)); mae = [summary[n]["right_arm_mae_deg"] for n in names]; rmse = [summary[n]["right_arm_rmse_deg"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained"); ax.bar(x-.18, mae, .36, label="MAE"); ax.bar(x+.18, rmse, .36, label="RMSE"); ax.set_xticks(x, labels); ax.set_ylabel("Right-arm joint error (deg)"); ax.set_title("Held-out post-closure error"); ax.grid(axis="y", alpha=.2); ax.legend(frameon=False); fig.savefig(ROOT / "metrics_comparison.png", dpi=170); plt.close(fig)

if __name__ == "__main__": main()
