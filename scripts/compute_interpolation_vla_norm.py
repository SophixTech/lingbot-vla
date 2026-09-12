#!/usr/bin/env python3
"""Compute VLA normalization stats directly from a LeRobot v3 parquet file.

This matches package_a2d's robot mapping without decoding video frames.  The
generic runner performs a costly per-sample Hugging Face selection; this
dataset has 749k frames, so direct columnar reads are required here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def statistics(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.percentile(values, 1, axis=0).tolist(),
        "q99": np.percentile(values, 99, axis=0).tolist(),
        "q02": np.percentile(values, 2, axis=0).tolist(),
        "q98": np.percentile(values, 98, axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    tables = []
    for path in sorted((args.dataset / "data").glob("chunk-*/*.parquet")):
        tables.append(pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"]))
    table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
    episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)

    if state.shape != action.shape or state.shape[1] != 16:
        raise ValueError(f"expected Nx16 matching state/action, got {state.shape} and {action.shape}")
    if np.any(np.diff(episode) < 0):
        raise ValueError("episode indices are not contiguous")

    arm_delta = np.empty((len(state), args.chunk_size, 14), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(episode)) + 1]
    ends = np.r_[starts[1:], len(state)]
    offsets = np.arange(args.chunk_size)
    for start, end in zip(starts, ends, strict=True):
        rows = np.arange(start, end)
        future = np.minimum(rows[:, None] + offsets, end - 1)
        arm_delta[start:end] = action[future, :14] - state[rows, None, :14]

    norm_stats = {
        "observation.state.arm.position": statistics(state[:, :14]),
        "observation.state.effector.position": statistics(state[:, 14:]),
        "action.arm.position": statistics(arm_delta.reshape(len(state), -1)),
        "action.effector.position": statistics(action[:, 14:]),
    }
    # LingBot's normalizer expects horizon-specific arm statistics.
    for key in ("mean", "std", "q01", "q99", "q02", "q98", "min", "max"):
        norm_stats["action.arm.position"][key] = np.asarray(
            norm_stats["action.arm.position"][key], dtype=np.float64
        ).reshape(args.chunk_size, 14).tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"norm_stats": norm_stats, "count": int(len(state))}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(state), "episodes": len(starts), "chunk_size": args.chunk_size}))


if __name__ == "__main__":
    main()
