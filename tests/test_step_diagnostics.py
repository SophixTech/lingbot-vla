import gc
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lingbotvla.utils.step_diagnostics import StepDiagnostics


class StepDiagnosticsTest(unittest.TestCase):
    def test_disabled_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'VLA_STEP_DIAG': '0'}):
            diag = StepDiagnostics(d, 0)
            self.assertIsNone(diag.file)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_rank_files_and_accumulated_stages(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'VLA_STEP_DIAG': '1'}):
            diag = StepDiagnostics(d, 3)
            diag.begin(12)
            diag.mark('data_wait')
            diag.mark('forward')
            first = diag.row['host_seconds']['forward']
            diag.mark('forward')
            self.assertGreaterEqual(diag.row['host_seconds']['forward'], first)
            diag.end(1.25)
            diag.flush()
            row = json.loads(next((Path(d) / 'step_diagnostics').glob('rank3-*.jsonl')).read_text())
            self.assertEqual(row['step'], 12)
            self.assertEqual(row['reported_step_s'], 1.25)
            self.assertIn('data_wait', row['host_seconds'])
            diag.close()

    def test_gc_timing_and_policy_unchanged(self):
        thresholds = gc.get_threshold()
        enabled = gc.isenabled()
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'VLA_STEP_DIAG': '1'}):
            diag = StepDiagnostics(d, 0)
            try:
                diag.begin(1)
                gc.collect(0)
                diag.mark('forward')
                diag.end(1.0)
                diag.flush()
                row = json.loads(next((Path(d) / 'step_diagnostics').glob('*.jsonl')).read_text())
                events = row['gc_events']
                self.assertTrue(events)
                self.assertTrue(any(e['generation'] == 0 for e in events))
                self.assertTrue(all(e['duration_s'] >= 0 for e in events))
                self.assertIn('forward', row['main_thread_cpu_seconds'])
                self.assertIn('ru_nivcsw', row['process_usage_delta'])
                self.assertEqual(row['stage_intervals'][0]['name'], 'forward')
                self.assertEqual(gc.get_threshold(), thresholds)
                self.assertEqual(gc.isenabled(), enabled)
            finally:
                diag.close()
            self.assertNotIn(diag._gc_callback, gc.callbacks)

    def test_fork_child_callback_is_ignored(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'VLA_STEP_DIAG': '1'}):
            diag = StepDiagnostics(d, 0)
            try:
                with patch('lingbotvla.utils.step_diagnostics.os.getpid', return_value=diag.pid + 1):
                    diag._gc_callback('start', {'generation': 2})
                    diag._gc_callback('stop', {'generation': 2})
                self.assertFalse(diag.gc_events)
                self.assertFalse(diag.gc_starts)
            finally:
                diag.close()
