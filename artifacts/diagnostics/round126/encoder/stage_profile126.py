"""R126 callback-only encoder clean timer and stage-probe scaffolding.

No Torch import, model/network/resource actions or executable main entry point.
The lead chooses the actual resident/staged encoder callbacks and device drain.
These helpers deliberately do not infer that an encode stage equals weight H2D.
Nested intervals are inclusive; synchronization probes are NOT clean performance.
"""
from __future__ import annotations

from contextlib import contextmanager
import time


class EncoderStageProbe126:
    """Host-dispatch or explicitly synchronized nested intervals, never both added.

    Wrap call SITES or use a context manager; do not stack unknown instance-method
    monkeypatches over the geometry cache's ownership-controlled lifecycle hooks.
    clean timers must leave this recorder disabled. Probe output copies/hashes are
    outside measure_complete(). Stage host wall can contain dispatch, native
    blocking copies and waits: it is not pure CPU or isolated device kernel time.
    """
    def __init__(self, *, mode='host_dispatch', drain=None, clock=time.perf_counter):
        if mode not in ('host_dispatch', 'synchronized_probe'):
            raise ValueError('Unknown stage-probe mode')
        if mode == 'synchronized_probe' and not callable(drain):
            raise ValueError('Synchronized probes require a lead-supplied device drain')
        self.mode, self.drain, self.clock = mode, drain, clock
        self.enabled = False
        self.rows = []
        self._stack = []
        self._serial = 0

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        if not isinstance(name, str) or not name:
            raise ValueError('A stage name is required')
        sync = self.mode == 'synchronized_probe'
        if sync:
            self.drain()
        started = self.clock()
        serial = self._serial
        self._serial += 1
        parent = self._stack[-1] if self._stack else None
        self._stack.append(serial)
        failed = True
        try:
            yield
            failed = False
        finally:
            try:
                if sync:
                    self.drain()
            finally:
                elapsed = self.clock() - started
                self._stack.pop()
                self.rows.append({'stage': name, 'id': serial, 'parent_id': parent,
                                  'inclusive_host_wall_s': elapsed, 'mode': self.mode,
                                  'failure': failed,
                                  'contains_child_intervals': any(r['parent_id'] == serial for r in self.rows),
                                  'pure_CPU_or_kernel_time_claimed': False})

    def call(self, name, function, *args, **kwargs):
        with self.stage(name):
            return function(*args, **kwargs)

    def reset(self):
        if self._stack:
            raise RuntimeError('Cannot reset while a stage is active')
        self.rows.clear()
        self._serial = 0

    def report(self):
        return {'mode': self.mode, 'rows': [dict(r) for r in self.rows],
                'clean_performance': False,
                'do_not_add_inclusive_parents_to_children': True,
                'unassigned_gaps_are_not_automatically_H2D_or_weight_offload': True}


def measure_complete(call, *, start_barrier, drain, clock=time.perf_counter, proof=None):
    """Complete callback wall; native hidden.cpu() stays inside call, proof outside.

    call must be the real unchanged R124 encode_ids/encode_prompt entry, including
    whatever load/offload/processor/broadcast that entry itself performs. Resident
    versus staged is the lead's explicit choice and must accompany every receipt.
    """
    start_barrier()
    drain()
    started = clock()
    output = call()
    drain()
    seconds = clock() - started
    # Do not put proof in the return expression before stopping the clock.
    evidence = None if proof is None else proof(output)
    return output, seconds, evidence
