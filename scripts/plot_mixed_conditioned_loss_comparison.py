"""Plot live mixed-conditioned loss against earlier LoRA runs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output")
SOURCES = {
    "mixed-conditioned (live)": ROOT / "separated_model/mixed_conditioned/checkpoints/loss.jsonl",
    "label-up": ROOT / "separated_model/label_up/checkpoints/loss.jsonl",
    "label-down": ROOT / "separated_model/label_down/checkpoints/loss.jsonl",
    "marked_9_1_0922": ROOT / "marked_9_1_0922/checkpoints/loss.jsonl",
    "marked_full0919": ROOT / "marked_full0919/checkpoints/loss.jsonl",
}
OUT = ROOT / "separated_model/mixed_conditioned/comparison"
WINDOW = 100


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def rolling(values: np.ndarray, window: int):
    if len(values) < window:
        return np.arange(1, len(values) + 1), values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window, len(values) + 1), smoothed


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = {name: load(path) for name, path in SOURCES.items()}
    colors = ["#1679b5", "#d65a19", "#3a9d5d", "#8a5cc2", "#bf3b53"]

    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    summary = {}
    for (name, rows), color in zip(datasets.items(), colors):
        steps = np.asarray([row["step"] for row in rows], dtype=int)
        losses = np.asarray([row["loss"] for row in rows], dtype=float)
        ax.plot(steps, losses, color=color, alpha=0.08, linewidth=0.4)
        smooth_steps, smooth = rolling(losses, WINDOW)
        ax.plot(smooth_steps, smooth, color=color, linewidth=1.8, label=name)
        tail = losses[-min(WINDOW, len(losses)):]
        summary[name] = {
            "last_step": int(steps[-1]),
            "last_loss": float(losses[-1]),
            "mean_last_100": float(tail.mean()),
            "min": float(losses.min()),
            "max": float(losses.max()),
            "nonfinite": int((~np.isfinite(losses)).sum()),
        }

    ax.set_title("Training loss comparison (raw + 100-step mean)")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Normalized L1 flow-matching loss")
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 0.45)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    path = OUT / "loss_comparison_live.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    (OUT / "loss_comparison_live_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"plot": str(path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
