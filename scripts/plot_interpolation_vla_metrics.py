#!/usr/bin/env python3
"""Render the current interpolation_vla training curve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output = Path(sys.argv[1])
    path = output / "checkpoints" / "loss.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = list({row["step"]: row for row in rows}.values())
    rows.sort(key=lambda row: row["step"])
    if not rows:
        return
    step = np.asarray([row["step"] for row in rows])
    loss = np.asarray([row["loss"] for row in rows])
    grad = np.asarray([row["grad_norm"] for row in rows])
    duration = np.asarray([row["step_time"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(step, loss, alpha=.45, label="step loss")
    if len(loss) >= 100:
        axes[0].plot(step[99:], np.convolve(loss, np.ones(100) / 100, mode="valid"), label="100-step mean")
    axes[0].set(title="VLA training loss", xlabel="step", ylabel="L1 flow-matching loss")
    axes[0].legend()
    axes[1].plot(step, grad)
    axes[1].set(title="Gradient norm", xlabel="step", ylabel="norm")
    axes[2].plot(step, duration)
    axes[2].set(title="Model step time", xlabel="step", ylabel="seconds")
    fig.savefig(output / "training_curves.png", dpi=160)


if __name__ == "__main__":
    main()
