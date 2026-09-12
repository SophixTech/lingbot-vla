#!/usr/bin/env python3
"""Export a loss curve from LingBot-VLA's checkpoint loss JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rolling_mean(values: list[float], window: int) -> list[float]:
    total = 0.0
    result: list[float] = []
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("loss_file", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.loss_file.read_text().splitlines() if line]
    if not records:
        raise SystemExit(f"No loss records found in {args.loss_file}")

    steps = [record["step"] for record in records]
    losses = [record["loss"] for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(steps, losses, color="#8fbce6", alpha=0.35, linewidth=0.8, label="step loss")
    axis.plot(
        steps,
        rolling_mean(losses, args.window),
        color="#1769aa",
        linewidth=2.0,
        label=f"{args.window}-step moving average",
    )
    axis.set(title="LingBot-VLA LoRA Post-Training Loss", xlabel="Training step", ylabel="L1 flow-matching loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
