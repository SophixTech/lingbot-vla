#!/usr/bin/env python3
"""Analyze a rollout diagnostic directory without touching deployment state."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _parse(value: str) -> Any:
    if value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if value in ("True", "False"):
            return value == "True"
        return value


def _load_rows(log_dir: Path) -> list[dict[str, Any]]:
    with (log_dir / "control_steps.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: _parse(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise RuntimeError("control_steps.csv contains no control steps")
    return rows


def _scalar(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.nan if row.get(key) is None else float(row[key]) for row in rows])


def _matrix(rows: list[dict[str, Any]], key: str) -> np.ndarray | None:
    width = next((len(value) for row in rows if isinstance((value := row.get(key)), list)), None)
    if width is None:
        return None
    result = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        value = row.get(key)
        if isinstance(value, list) and len(value) == width:
            result[index] = value
    return result


def _stats(values: np.ndarray | None, p50: bool = True) -> dict[str, float | None]:
    finite = np.asarray([]) if values is None else np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite)]
    result = {"mean": None, "p95": None, "max": None}
    if p50:
        result["p50"] = None
    if finite.size:
        result.update(mean=float(np.mean(finite)), p95=float(np.percentile(finite, 95)), max=float(np.max(finite)))
        if p50:
            result["p50"] = float(np.percentile(finite, 50))
    return result


def _mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if not finite.size else float(np.mean(finite))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator / max(denominator, 1e-12))


def _derive(rows: list[dict[str, Any]], chunks: list[np.ndarray], fps: float) -> dict[str, Any]:
    actions = _matrix(rows, "executed_action")
    q_cmd, q_actual = _matrix(rows, "q_cmd"), _matrix(rows, "q_actual")
    starts = np.asarray([bool(row.get("is_chunk_start")) for row in rows])
    latency = _scalar(rows, "inference_latency_ms")
    # Latency repeats for every step in a chunk; report one real request per chunk.
    latency = latency[starts & np.isfinite(latency)]
    age = _scalar(rows, "observation_age_at_execution_ms")
    gaps = _scalar(rows, "command_gap_ms")
    boundary_logged = _scalar(rows, "boundary_delta_l2")
    velocity, acceleration, jerk = (_matrix(rows, key) for key in ("velocity", "acceleration", "jerk"))
    chunk_boundary_deltas = np.asarray([
        np.linalg.norm(chunk[0] - previous[-1])
        for previous, chunk in zip(chunks, chunks[1:])
        if previous.ndim == 2 and chunk.ndim == 2 and len(previous) and len(chunk)
    ], dtype=float)
    chunk_velocity_deltas = np.asarray([
        np.linalg.norm((chunk[1] - chunk[0]) - (previous[-1] - previous[-2]))
        for previous, chunk in zip(chunks, chunks[1:])
        if previous.ndim == 2 and chunk.ndim == 2 and len(previous) >= 2 and len(chunk) >= 2
    ], dtype=float)

    action_delta = np.linalg.norm(np.diff(actions, axis=0), axis=1) if actions is not None and len(actions) > 1 else np.asarray([])
    delta_starts = starts[1:] if action_delta.size else np.asarray([], dtype=bool)
    intra_delta = action_delta[~delta_starts]
    boundary_delta = boundary_logged[np.isfinite(boundary_logged)]
    if not boundary_delta.size and action_delta.size:
        boundary_delta = action_delta[delta_starts]
    # Gripper commands are intentionally quantized/clipped at the hardware
    # boundary. Use arm dimensions for the VLA smoothness diagnosis so a
    # legitimate open/close transition cannot masquerade as arm chatter.
    jerk_norm = np.linalg.norm(jerk[:, :14], axis=1) if jerk is not None and jerk.shape[1] >= 14 else (np.linalg.norm(jerk, axis=1) if jerk is not None else np.full(len(rows), np.nan))
    valid_jerk = np.isfinite(jerk_norm)
    near_boundary = starts.copy()
    near_boundary[:-1] |= starts[1:]
    near_boundary[1:] |= starts[:-1]
    inside_jerk = jerk_norm[valid_jerk & ~near_boundary]
    boundary_jerk = jerk_norm[valid_jerk & near_boundary]
    intra_mean, boundary_mean = _mean(intra_delta), _mean(boundary_delta)
    inside_jerk_mean, boundary_jerk_mean = _mean(inside_jerk), _mean(boundary_jerk)

    peak_alignment = None
    finite_jerk = jerk_norm[valid_jerk]
    if finite_jerk.size >= 10 and np.any(near_boundary & valid_jerk):
        threshold = np.percentile(finite_jerk, 95)
        peaks = valid_jerk & (jerk_norm >= threshold)
        peak_alignment = float(np.mean(near_boundary[peaks])) if np.any(peaks) else 0.0

    tracking = None if q_cmd is None or q_actual is None else q_cmd - q_actual
    command_hf = actual_hf = None
    if q_cmd is not None and q_actual is not None and len(q_cmd) >= 3:
        command_hf = float(np.nanmean(np.abs(np.diff(q_cmd, n=2, axis=0))))
        actual_hf = float(np.nanmean(np.abs(np.diff(q_actual, n=2, axis=0))))
    tracking_hf_ratio = _ratio(actual_hf, command_hf)
    period_ms = 1000.0 / fps
    boundary_gap = gaps[starts & np.isfinite(gaps)]
    max_boundary_gap = None if not boundary_gap.size else float(np.max(boundary_gap))

    action_ratio = _ratio(boundary_mean, intra_mean)
    jerk_ratio = _ratio(boundary_jerk_mean, inside_jerk_mean)
    type1 = action_ratio is not None and jerk_ratio is not None and peak_alignment is not None and action_ratio >= 2 and jerk_ratio >= 2 and peak_alignment >= .5
    type2 = max_boundary_gap is not None and max_boundary_gap >= period_ms and latency.size > 0
    type3 = inside_jerk.size >= 10 and boundary_jerk.size >= 5 and np.percentile(inside_jerk, 95) >= np.percentile(boundary_jerk, 95) and np.percentile(inside_jerk, 95) > 0
    type4 = tracking_hf_ratio is not None and tracking_hf_ratio >= 3 and _stats(tracking, False)["p95"] not in (None, 0.0)

    def result(detected: bool, available: bool, evidence: dict[str, Any]) -> dict[str, Any]:
        return {"status": "detected" if detected else ("not_detected" if available else "insufficient_data"), "evidence": evidence}

    diagnoses = {
        "type_1_chunk_boundary_discontinuity": result(type1, action_ratio is not None and jerk_ratio is not None and peak_alignment is not None, {"boundary_intra_action_ratio": action_ratio, "boundary_jerk_ratio": jerk_ratio, "jerk_peak_boundary_alignment": peak_alignment}),
        "type_2_inference_latency_or_blocking": result(type2, max_boundary_gap is not None and latency.size > 0, {"max_boundary_command_gap_ms": max_boundary_gap, "inference_latency_p95_ms": _stats(latency)["p95"], "control_period_ms": period_ms}),
        "type_3_vla_output_not_smooth": result(type3, inside_jerk.size >= 10 and boundary_jerk.size > 0, {"inside_chunk_jerk_p95": _stats(inside_jerk)["p95"], "boundary_neighborhood_jerk_p50": _stats(boundary_jerk)["p50"]}),
        "type_4_low_level_tracking_oscillation": result(type4, tracking_hf_ratio is not None, {"actual_command_high_frequency_ratio": tracking_hf_ratio, "tracking_error_p95": _stats(tracking, False)["p95"]}),
    }
    detected = [name for name, value in diagnoses.items() if value["status"] == "detected"]
    classification = "Type 5: mixed" if len(detected) > 1 else (detected[0] if detected else "No supported type detected")
    return {
        "inference_latency_ms": _stats(latency),
        "observation_age_ms": _stats(age),
        "mean_intra_chunk_action_delta": intra_mean,
        "mean_boundary_action_delta": boundary_mean,
        "mean_full_chunk_tail_to_next_prefix_delta": _mean(chunk_boundary_deltas),
        "mean_boundary_velocity_delta": _mean(chunk_velocity_deltas),
        "boundary_intra_action_ratio": action_ratio,
        "mean_jerk_inside_chunk": inside_jerk_mean,
        "mean_jerk_near_chunk_boundary": boundary_jerk_mean,
        "boundary_jerk_ratio": jerk_ratio,
        "jerk_peak_boundary_alignment": peak_alignment,
        "command_tracking_error": _stats(tracking, False),
        "chunks_saved": len(chunks),
        "classification": classification,
        "diagnoses": diagnoses,
    }


def _boundaries(ax, times: np.ndarray, starts: np.ndarray) -> None:
    for value in times[starts]:
        ax.axvline(value, color="tab:red", alpha=.25, linewidth=.8)


def _save_fig(path: Path, title: str, times: np.ndarray, values: np.ndarray | None, starts: np.ndarray, ylabel: str = "") -> None:
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    if values is not None:
        ax.plot(times, values, linewidth=.7)
    _boundaries(ax, times, starts)
    ax.set(title=title, xlabel="monotonic seconds", ylabel=ylabel)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plots(log_dir: Path, rows: list[dict[str, Any]]) -> None:
    times = _scalar(rows, "timestamp")
    starts = np.asarray([bool(row.get("is_chunk_start")) for row in rows])
    action, q_cmd, q_actual = (_matrix(rows, key) for key in ("executed_action", "q_cmd", "q_actual"))
    velocity, acceleration, jerk = (_matrix(rows, key) for key in ("velocity", "acceleration", "jerk"))

    if q_cmd is not None and q_actual is not None:
        dims = q_cmd.shape[1]
        fig, axes = plt.subplots((dims + 1) // 2, 2, figsize=(15, 2.3 * ((dims + 1) // 2)), sharex=True, constrained_layout=True)
        for index, ax in enumerate(np.asarray(axes).reshape(-1)):
            if index >= dims:
                ax.axis("off"); continue
            ax.plot(times, q_cmd[:, index], label="command", linewidth=.8)
            ax.plot(times, q_actual[:, index], label="actual", linewidth=.8)
            _boundaries(ax, times, starts); ax.set_title(f"joint {index}")
        axes.flat[0].legend()
        fig.savefig(log_dir / "command_vs_actual.png", dpi=150); plt.close(fig)
    _save_fig(log_dir / "action_with_chunk_boundaries.png", "Executed action and chunk boundaries", times, action, starts)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ax, values, title in zip(axes, (velocity, acceleration, jerk), ("Velocity", "Acceleration", "Jerk")):
        if values is not None: ax.plot(times, values, linewidth=.55)
        _boundaries(ax, times, starts); ax.set_title(title)
    fig.savefig(log_dir / "velocity_acceleration_jerk.png", dpi=150); plt.close(fig)
    for key, filename, title in (("boundary_delta_l2", "boundary_delta_l2.png", "Boundary delta L2"), ("inference_latency_ms", "inference_latency.png", "Inference latency (ms)"), ("observation_age_at_execution_ms", "observation_age.png", "Observation age at execution (ms)")):
        _save_fig(log_dir / filename, title, times, _scalar(rows, key), starts)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    seen: set[int] = set()
    for row in rows:
        chunk = int(row.get("chunk_id") or 0)
        if chunk not in seen and row.get("inference_start_time") is not None and row.get("inference_end_time") is not None:
            ax.plot([row["inference_start_time"], row["inference_end_time"]], [chunk, chunk], color="tab:orange", linewidth=4)
            seen.add(chunk)
        if row.get("action_execution_time") is not None: ax.plot(row["action_execution_time"], chunk, "k.", markersize=2)
    ax.set(title="Inference intervals (orange) and action executions (black)", xlabel="monotonic seconds", ylabel="chunk id")
    fig.savefig(log_dir / "execution_inference_timeline.png", dpi=150); plt.close(fig)

    jerk_norm = None if jerk is None else np.linalg.norm(jerk, axis=1)
    _save_fig(log_dir / "jerk_boundary_alignment.png", "Jerk peaks and chunk boundaries", times, jerk_norm, starts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.log_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = _load_rows(args.log_dir)
    chunks = [np.load(path, allow_pickle=False) for path in sorted(args.log_dir.glob("chunk_*.npy"))]
    result = _derive(rows, chunks, float(metadata.get("training_fps", 30.0)))
    (args.log_dir / "jitter_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _plots(args.log_dir, rows)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
