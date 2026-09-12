from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_rollout_jitter as analyze
from rollout_logger import RolloutLogger


class RolloutJitterTests(unittest.TestCase):
    def test_full_chunks_and_executed_boundary_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "rollout"
            logger = RolloutLogger(root, {"training_fps": 30})
            first = np.zeros((50, 3), dtype=np.float32)
            second = np.full((50, 3), 4.0, dtype=np.float32)
            logger.record_chunk(0, first)
            logger.record_control_step({"chunk_id": 0, "is_chunk_start": True, "executed_action": [2, 2, 2], "action_execution_time": 0})
            logger.record_chunk(1, second)
            logger.record_control_step({"chunk_id": 1, "is_chunk_start": True, "executed_action": [5, 5, 5], "action_execution_time": 1 / 30})
            logger.close()

            rows = analyze._load_rows(root)
            self.assertIs(rows[0]["is_chunk_start"], True)
            self.assertAlmostEqual(rows[1]["boundary_delta_l2"], np.sqrt(12))
            np.testing.assert_array_equal(np.load(root / "chunk_001.npy"), second)

    def test_missing_actual_state_stays_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "rollout"
            logger = RolloutLogger(root, {"training_fps": 30})
            logger.record_chunk(0, np.zeros((50, 2), dtype=np.float32))
            for index in range(12):
                logger.record_control_step({"chunk_id": 0, "step_in_chunk": index,
                                            "is_chunk_start": index == 0,
                                            "executed_action": [index, index],
                                            "action_execution_time": index / 30})
            logger.close()
            rows = analyze._load_rows(root)
            result = analyze._derive(rows, [np.load(root / "chunk_000.npy")], 30)
            self.assertEqual(result["diagnoses"]["type_4_low_level_tracking_oscillation"]["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
