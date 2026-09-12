#!/usr/bin/env python3
"""Compare current LingBot-VLA loss against a completed reference run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    by_step = {
        row["step"]: row
        for line in path.read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    rows = [by_step[step] for step in sorted(by_step)]
    return (
        np.asarray([row["step"] for row in rows]),
        np.asarray([row["loss"] for row in rows]),
    )


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    ref_step, ref_loss = load_loss(args.reference)
    cur_step, cur_loss = load_loss(args.current)
    window = min(args.window, len(ref_loss), len(cur_loss))
    shared_end = min(ref_step[-1], cur_step[-1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
    for axis, step, loss, label, color in (
        (axes[0], ref_step, ref_loss, "previous package VLA", "#4C78A8"),
        (axes[0], cur_step, cur_loss, "interpolation VLA", "#E45756"),
    ):
        axis.plot(
            step[window - 1 :], moving_mean(loss, window), color=color, linewidth=2, label=label
        )
    axes[0].axvline(shared_end, color="#666666", linestyle="--", linewidth=1)
    axes[0].set(
        title=f"100-step training-loss comparison through step {shared_end}",
        xlabel="training step",
        ylabel="L1 flow-matching loss",
        xlim=(0, shared_end),
        ylim=(0.08, 0.8),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    ref_at_current = np.interp(cur_step, ref_step, ref_loss)
    gap = moving_mean(cur_loss - ref_at_current, window)
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].plot(cur_step[window - 1 :], gap, color="#E45756", linewidth=2)
    axes[1].fill_between(
        cur_step[window - 1 :], gap, 0, where=gap <= 0, color="#54A24B", alpha=0.25,
        label="current lower (better)",
    )
    axes[1].fill_between(
        cur_step[window - 1 :], gap, 0, where=gap > 0, color="#E45756", alpha=0.2,
        label="current higher",
    )
    axes[1].set(
        title="Interpolation minus previous run (same step)",
        xlabel="training step",
        ylabel="100-step mean loss difference",
        xlim=(0, shared_end),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
