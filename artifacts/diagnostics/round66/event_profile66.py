"""Preallocated same-current-stream event pairs; no per-component synchronize.

Event elapsed time can contain stream stalls, host dispatch gaps, dependencies
and waits; it is not an isolated kernel occupancy measurement. CPU dispatch wall
is a different quantity, and nested categories must never be added to parents.
"""
import time


class EventProfile:
    def __init__(self, torch_module, capacity=1400):
        self.torch = torch_module
        self.pairs = [(torch_module.npu.Event(enable_timing=True), torch_module.npu.Event(enable_timing=True)) for _ in range(capacity)]
        # Force lazy event handles/record paths before any measured forward.
        for start, end in self.pairs:
            start.record()
            end.record()
        torch_module.npu.synchronize()
        self.active = False
        self.records = []

    def begin(self):
        if self.active:
            raise RuntimeError('A component profile is already active')
        self.records = []
        self.stream_begin = str(self.torch.npu.current_stream())
        self.active = True

    def call(self, name, function, *args, **kwargs):
        if not self.active:
            return function(*args, **kwargs)
        index = len(self.records)
        if index >= len(self.pairs):
            raise RuntimeError('Preallocated profile event budget exceeded')
        start, end = self.pairs[index]
        row = {'name': name, 'pair': index}
        # Reserve before entering so nested calls never reuse the parent pair.
        self.records.append(row)
        before = time.perf_counter()
        start.record()
        try:
            return function(*args, **kwargs)
        finally:
            end.record()
            row['host_dispatch_s'] = time.perf_counter() - before

    def finish(self):
        if not self.active:
            raise RuntimeError('Profile is not active')
        # Only one drain after the COMPLETE forward. This also ensures all event
        # timestamps are ready; no event.synchronize() in the component loop.
        self.torch.npu.synchronize()
        self.stream_end = str(self.torch.npu.current_stream())
        if self.stream_begin != self.stream_end:
            raise RuntimeError('Current stream changed across profiled forward')
        self.active = False
        out = []
        for record in self.records:
            start, end = self.pairs[record['pair']]
            out.append({'name': record['name'], 'current_stream_elapsed_ms': start.elapsed_time(end),
                        'host_dispatch_s': record['host_dispatch_s']})
        return out
