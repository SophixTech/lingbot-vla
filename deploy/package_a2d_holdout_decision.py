#!/usr/bin/env python3
"""Make a paired, bootstrap-backed checkpoint decision from holdout episodes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_metrics(path: Path) -> dict[int, float]:
    with path.open(newline="") as metrics_file:
        reader = csv.DictReader(metrics_file, delimiter="\t")
        return {int(row["episode_index"]): float(row["mae"]) for row in reader}


def bootstrap_interval(deltas: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(10_000, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    args = parser.parse_args()

    current = read_metrics(args.metrics)
    if len(current) < 20:
        raise SystemExit(f"expected at least 20 holdout episodes, got {len(current)}")
    current_mae = float(np.mean(list(current.values())))

    if not args.history.exists():
        print(f"{len(current)}\t{current_mae:.12g}\tbaseline\t\t\t")
        return

    with args.history.open(newline="") as history_file:
        rows = list(csv.DictReader(history_file, delimiter="\t"))
    if not rows:
        print(f"{len(current)}\t{current_mae:.12g}\tbaseline\t\t\t")
        return

    best = min(rows, key=lambda row: float(row["mae"]))
    best_step = int(best["step"])
    best_metrics = args.metrics.parent.parent / f"open_loop_holdout_step{best_step}" / "per_episode_metrics.tsv"
    reference = read_metrics(best_metrics)
    common = sorted(set(current) & set(reference))
    if len(common) < 20:
        raise SystemExit(f"need at least 20 paired holdout episodes, got {len(common)}")

    # Positive deltas mean the current checkpoint lowers per-episode MAE.
    deltas = np.array([reference[episode] - current[episode] for episode in common])
    relative = float(deltas.mean() / np.mean([reference[episode] for episode in common]))
    ci_low, ci_high = bootstrap_interval(deltas, seed=args.step)
    improved = relative >= args.min_relative_improvement and ci_low > 0
    decision = "improved" if improved else "not_improved"
    print(f"{len(common)}\t{current_mae:.12g}\t{decision}\t{relative:.12g}\t{ci_low:.12g}\t{ci_high:.12g}")


if __name__ == "__main__":
    main()
