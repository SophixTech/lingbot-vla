#!/usr/bin/env python3
"""Plot the shared training prefix for the 10- and 20-step VLA runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/bjtc/Sophix/lingbot-vla/output")
OLD = ROOT / "interpolation_vla/checkpoints/loss.jsonl"
NEW = ROOT / "interpolation_vla_num_denoising_step_20/checkpoints/loss.jsonl"
OUT = ROOT / "interpolation_vla_num_denoising_step_20/vs_interpolation_vla_shared_prefix.png"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def moving_average(values: np.ndarray, window: int = 50) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


old, new = load(OLD), load(NEW)
n = min(len(old), len(new))
old, new = old[:n], new[:n]
x = np.arange(1, n + 1)
old_loss = np.array([row["loss"] for row in old])
new_loss = np.array([row["loss"] for row in new])
old_grad = np.array([row["grad_norm"] for row in old])
new_grad = np.array([row["grad_norm"] for row in new])
window = min(50, n)
smooth_x = x[window - 1 :]

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, layout="constrained")

axes[0].plot(x, old_loss, color="#1677ff", alpha=0.22, linewidth=0.8, label="Previous (10 steps), raw")
axes[0].plot(x, new_loss, color="#f5222d", alpha=0.22, linewidth=0.8, label="Current (20 steps), raw")
axes[0].plot(smooth_x, moving_average(old_loss, window), color="#0b4a9e", linewidth=2, label=f"Previous {window}-step mean")
axes[0].plot(smooth_x, moving_average(new_loss, window), color="#a8071a", linewidth=1.2, linestyle="--", label=f"Current {window}-step mean")
axes[0].set_ylabel("Training loss (L1 flow matching)")
axes[0].set_title(f"Same training trajectory through shared prefix: {n} steps")
axes[0].set_ylim(0, min(max(np.percentile(np.r_[old_loss, new_loss], 99.5) * 1.1, 0.6), 2.0))
axes[0].legend(ncol=2, fontsize=9)

axes[1].plot(x, old_grad, color="#1677ff", linewidth=1.2, label="Previous (10 steps)")
axes[1].plot(x, new_grad, color="#f5222d", linewidth=0.9, linestyle="--", label="Current (20 steps)")
axes[1].set_ylabel("Gradient norm")
axes[1].legend()

loss_diff = new_loss - old_loss
grad_diff = new_grad - old_grad
axes[2].plot(x, loss_diff, color="#722ed1", linewidth=1, label="Loss: current minus previous")
axes[2].plot(x, grad_diff, color="#13c2c2", linewidth=1, label="Grad norm: current minus previous")
axes[2].axhline(0, color="#333333", linewidth=0.8)
axes[2].set_xlabel("Training step")
axes[2].set_ylabel("Numeric difference")
axes[2].legend()

fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"shared_steps={n}")
print(f"max_abs_loss_difference={np.abs(loss_diff).max():.12g}")
print(f"max_abs_grad_difference={np.abs(grad_diff).max():.12g}")
print(OUT)
