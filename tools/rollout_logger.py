"""Low-overhead, opt-in rollout diagnostics for the G1 VLA client.

The logger only serializes observations already available at the deployment
boundary.  It never transforms an action or participates in control timing.
When disabled (``logger=None``), callers retain the original code path.
"""
from __future__ import annotations

import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTROL_COLUMNS = (
    "timestamp", "wall_time_ns", "control_step", "chunk_id", "step_in_chunk",
    "is_chunk_start", "observation_capture_time", "inference_request_time",
    "inference_start_time", "inference_end_time", "action_execution_time",
    "inference_latency_ms", "observation_age_at_execution_ms", "command_gap_ms",
    "executed_action", "q_cmd", "q_actual", "dq_cmd", "dq_actual", "ee_cmd",
    "ee_actual", "gripper_cmd", "gripper_actual", "torque_cmd", "torque_actual",
    "kp", "kd", "boundary_delta", "boundary_delta_l2", "velocity_delta",
    "velocity", "acceleration", "jerk", "metadata",
)


def _json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (bool, np.bool_, list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    return str(value)


class RolloutLogger:
    """Write one self-contained diagnostic directory per rollout."""

    def __init__(self, root: str | Path, metadata: Mapping[str, Any] | None = None) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        self.root = root
        self._control_file = (root / "control_steps.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._control_file, fieldnames=CONTROL_COLUMNS)
        self._writer.writeheader()
        payload = {"schema_version": 1, "created_wall_time_ns": time.time_ns(), **(metadata or {})}
        (root / "metadata.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        self._period_s = 1.0 / float(payload.get("training_fps", 30.0))
        self._queue: queue.Queue = queue.Queue(maxsize=8192)
        self._error: Exception | None = None
        self._previous_action: np.ndarray | None = None
        self._previous_velocity: np.ndarray | None = None
        self._previous_acceleration: np.ndarray | None = None
        self._previous_execution_time: float | None = None
        self._previous_chunk: np.ndarray | None = None
        self._pending_chunk_metrics: dict[int, dict[str, Any]] = {}
        self._thread = threading.Thread(target=self._write_loop, name="rollout-diagnostics", daemon=True)
        self._thread.start()
        self._closed = False

    def record_chunk(self, chunk_id: int, action: Any, **metadata: Any) -> Path:
        array = np.asarray(action, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("action chunk must be a finite 2-D array")
        path = self.root / f"chunk_{int(chunk_id):03d}.npy"
        metrics: dict[str, Any] = {}
        if len(array):
            metrics["first_action"] = array[0].copy()
        if self._previous_chunk is not None and len(self._previous_chunk) >= 2 and len(array) >= 2:
            metrics["velocity_delta"] = (array[1] - array[0]) - (self._previous_chunk[-1] - self._previous_chunk[-2])
        self._pending_chunk_metrics[int(chunk_id)] = metrics
        self._previous_chunk = array.copy()
        self._put(("chunk", path, array.copy(), {"chunk_id": int(chunk_id), **metadata}))
        return path

    def record_control_step(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("rollout logger is closed")
        enriched = dict(record)
        chunk_metrics: dict[str, Any] = {}
        if bool(enriched.get("is_chunk_start")) and enriched.get("chunk_id") is not None:
            chunk_metrics = self._pending_chunk_metrics.pop(int(enriched["chunk_id"]), {})
            for key, value in chunk_metrics.items():
                if key == "first_action":
                    continue
                enriched.setdefault(key, value)
        action_value = enriched.get("executed_action")
        action = None if action_value is None else np.asarray(action_value, dtype=float)
        if action is not None and self._previous_action is not None:
            velocity = action - self._previous_action
            enriched.setdefault("velocity", velocity)
            if bool(enriched.get("is_chunk_start")):
                boundary_delta = np.asarray(chunk_metrics.get("first_action", action)) - self._previous_action
                enriched.setdefault("boundary_delta", boundary_delta)
                enriched.setdefault("boundary_delta_l2", float(np.linalg.norm(boundary_delta)))
            if self._previous_velocity is not None:
                acceleration = velocity - self._previous_velocity
                enriched.setdefault("acceleration", acceleration)
                if self._previous_acceleration is not None:
                    enriched.setdefault("jerk", acceleration - self._previous_acceleration)
                self._previous_acceleration = acceleration
            self._previous_velocity = velocity
        if action is not None:
            self._previous_action = action.copy()
        execution_time = enriched.get("action_execution_time")
        if execution_time is not None and self._previous_execution_time is not None:
            enriched.setdefault("command_gap_ms", max(0.0, (float(execution_time) - self._previous_execution_time - self._period_s) * 1000))
        if execution_time is not None:
            self._previous_execution_time = float(execution_time)
        row = {column: _json_value(enriched.get(column)) for column in CONTROL_COLUMNS}
        self._put(("row", row))

    def record_event(self, event: str, **fields: Any) -> None:
        self._put(("event", {"event": event, "wall_time_ns": time.time_ns(), **fields}))

    def _put(self, item: tuple) -> None:
        if self._error is not None:
            raise RuntimeError("rollout diagnostic writer failed") from self._error
        self._queue.put_nowait(item)

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if item[0] == "row":
                    self._writer.writerow(item[1])
                elif item[0] == "chunk":
                    _, path, array, metadata = item
                    np.save(path, array, allow_pickle=False)
                    with (self.root / "chunk_metadata.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(metadata, ensure_ascii=True) + "\n")
                elif item[0] == "event":
                    with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(item[1], ensure_ascii=True) + "\n")
        except Exception as exc:
            self._error = exc
        finally:
            self._control_file.flush()

    def close(self) -> None:
        if not self._closed:
            self._queue.put(None, timeout=1.0)
            self._thread.join(timeout=5.0)
            self._control_file.close()
            self._closed = True
            if self._thread.is_alive() or self._error is not None:
                raise RuntimeError("rollout diagnostics could not be flushed") from self._error

    def __enter__(self) -> "RolloutLogger":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
