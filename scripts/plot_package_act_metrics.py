#!/usr/bin/env python3
"""Render current ACT training metrics as a local PNG after each monitor pass."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    output = Path(sys.argv[1])
    metrics = output / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()] if metrics.exists() else []
    # A resumed run appends replacement metrics for steps that were reached
    # before an interruption. Keep the newest value at each step so the plot
    # represents the actual resumed trajectory instead of drawing it twice.
    def latest_per_step(event: str) -> list[dict]:
        by_step = {row["step"]: row for row in rows if row.get("event") == event}
        return [by_step[step] for step in sorted(by_step)]

    train = latest_per_step("train")
    validation = latest_per_step("validation")
    if not train:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    steps = [row["step"] for row in train]
    axes[0, 0].plot(steps, [row["loss"] for row in train], label="total loss")
    axes[0, 0].plot(steps, [row.get("l1_loss") for row in train], label="L1 action")
    axes[0, 0].plot(steps, [row.get("kld_loss") for row in train], label="KL")
    axes[0, 0].set(title="Training objective", xlabel="step", ylabel="loss")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, [row["grad_norm"] for row in train])
    axes[0, 1].set(title="Gradient norm", xlabel="step", ylabel="norm")
    axes[1, 0].plot(steps, [row["steps_per_s"] for row in train])
    axes[1, 0].set(title="Reported throughput (resets after resume)", xlabel="step", ylabel="steps / s")
    if validation:
        axes[1, 1].plot([row["step"] for row in validation], [row["action_mae_normalized"] for row in validation], marker="o")
        axes[1, 1].set(title="Held-out action MAE", xlabel="step", ylabel="normalized MAE")
    else:
        axes[1, 1].text(.5, .5, "First held-out evaluation at step 1,000", ha="center", va="center")
        axes[1, 1].set_axis_off()
    fig.savefig(output / "training_curves.png", dpi=160)


if __name__ == "__main__":
    main()
