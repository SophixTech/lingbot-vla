"""CPU-only deployment-side alignment and delay checks."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _rt():
    path = Path(__file__).parents[2] / "g1" / "运控test" / "g1_realtime_chunking.py"
    spec = importlib.util.spec_from_file_location("g1_realtime_chunking_contract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_latency_tracker_uses_ceil_and_recent_max():
    rt = _rt()
    tracker = rt.LatencyTracker(initial_delay_steps=0)
    tracker.observe(0.69)
    assert tracker.estimated_delay_steps == 21
    tracker.observe(0.74)
    assert tracker.estimated_delay_steps == 23
    for _ in range(10):
        tracker.observe(0.1)
    assert tracker.estimated_delay_steps == 3


def test_previous_chunk_context_is_timeline_aligned():
    rt = _rt()
    action = np.arange(50 * 16, dtype=np.float32).reshape(50, 16)
    model = rt.Prediction(7, 0.0, 0.0, 0.0, 0.0, np.zeros(16, np.float32), action,
                          {}, model_action=action.copy())
    timeline = rt.ActionTimeline()
    timeline.install(model, 0.0)
    context = timeline.rtc_context(15, 10)
    np.testing.assert_array_equal(context["prev_chunk_left_over"], action[15:])
    assert context["source_chunk_id"] == 7
    assert context["source_step_idx"] == 15


def test_padded_model_action_is_kept_for_rtc():
    rt = _rt()
    action = np.zeros((50, 16), dtype=np.float32)
    model_action = np.arange(50 * 75, dtype=np.float32).reshape(50, 75)
    prediction = rt.Prediction(1, 0.0, 0.0, 0.0, 0.0, np.zeros(16, np.float32), action,
                               {}, model_action=model_action)
    timeline = rt.ActionTimeline()
    timeline.install(prediction, 0.0)
    context = timeline.rtc_context(15, 10)
    np.testing.assert_array_equal(context["prev_chunk_left_over"], model_action[15:])
