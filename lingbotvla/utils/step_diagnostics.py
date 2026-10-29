"""Opt-in host diagnostics; no CUDA synchronization, collectives, or GC policy changes."""
import atexit
from collections import deque
import gc
import json
import os
import resource
import time
import warnings
from pathlib import Path


class StepDiagnostics:
    def __init__(self, output_dir, rank):
        self.file = None
        self.rows = []
        self.pid = os.getpid()
        self.rank = rank
        self.gc_events = deque(maxlen=512)
        self.gc_dropped = 0
        self.gc_starts = {}
        self.previous_end = None
        self.previous_flush_s = 0.0
        if os.environ.get('VLA_STEP_DIAG') == '1':
            try:
                directory = Path(output_dir) / 'step_diagnostics'
                directory.mkdir(parents=True, exist_ok=True)
                self.file = (directory / f'rank{rank}-pid{self.pid}.jsonl').open('a')
                gc.callbacks.append(self._gc_callback)
                atexit.register(self.close)
            except OSError as exc:
                warnings.warn(f'Step diagnostics disabled: {exc}')

    def _gc_callback(self, phase, info):
        # DataLoader fork children must not log into the parent process file.
        if os.getpid() != self.pid or self.file is None:
            return
        generation = info['generation']
        now = time.perf_counter()
        if phase == 'start':
            self.gc_starts[generation] = now
        elif phase == 'stop':
            started = self.gc_starts.pop(generation, None)
            if started is not None:
                if len(self.gc_events) == self.gc_events.maxlen:
                    self.gc_dropped += 1
                self.gc_events.append({
                    'generation': generation, 'start_monotonic_s': started,
                    'end_monotonic_s': now, 'duration_s': now - started,
                    'collected': info.get('collected'),
                    'uncollectable': info.get('uncollectable'),
                })

    def begin(self, step):
        self.started = self.last = time.perf_counter()
        self.last_thread_cpu = time.thread_time()
        self.usage = resource.getrusage(resource.RUSAGE_SELF)
        self.row = {
            'schema_version': 2, 'rank': self.rank, 'pid': self.pid,
            'step': step, 'wall_time': time.time(), 'start_monotonic_s': self.started,
            'host_seconds': {}, 'main_thread_cpu_seconds': {}, 'stage_intervals': [],
            'previous_flush_s': self.previous_flush_s,
        }
        self.previous_flush_s = 0.0
        if self.previous_end is not None:
            self.row['between_steps_s'] = self.started - self.previous_end

    def mark(self, name):
        now = time.perf_counter()
        cpu = time.thread_time()
        stages = self.row['host_seconds']
        stages[name] = stages.get(name, 0.0) + now - self.last
        cpus = self.row['main_thread_cpu_seconds']
        cpus[name] = cpus.get(name, 0.0) + cpu - self.last_thread_cpu
        self.row['stage_intervals'].append({
            'name': name, 'start_offset_s': self.last - self.started,
            'end_offset_s': now - self.started,
        })
        self.last, self.last_thread_cpu = now, cpu

    def end(self, reported_step_s):
        self.mark('logging_visualization_checkpoint')
        self.row['reported_step_s'] = reported_step_s
        self.row['iteration_s'] = time.perf_counter() - self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.row['process_usage_delta'] = {
            key: getattr(usage, key) - getattr(self.usage, key)
            for key in ('ru_utime', 'ru_stime', 'ru_minflt', 'ru_majflt', 'ru_nvcsw', 'ru_nivcsw')
        }
        # Events can precede this step (epoch setup or previous log flush).
        events = []
        while self.gc_events:
            events.append(self.gc_events.popleft())
        self.row['gc_events'] = events
        self.row['gc_events_dropped'] = self.gc_dropped
        self.gc_dropped = 0
        self.rows.append(self.row)
        if len(self.rows) >= 32:
            self.flush()
        self.previous_end = time.perf_counter()

    def flush(self):
        if os.getpid() != self.pid:
            return
        if self.file is not None and self.rows:
            started = time.perf_counter()
            try:
                self.file.write(''.join(json.dumps(row) + '\n' for row in self.rows))
                self.file.flush()
            except OSError as exc:
                warnings.warn(f'Step diagnostics disabled: {exc}')
                try:
                    self.file.close()
                except OSError:
                    pass
                self.file = None
                self._detach()
            self.rows.clear()
            self.previous_flush_s += time.perf_counter() - started

    def _detach(self):
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)

    def close(self):
        if os.getpid() != self.pid:
            return
        self._detach()
        self.flush()
        if self.file is not None:
            self.file.close()
            self.file = None
        atexit.unregister(self.close)
