"""Private owner-thread NPU stream capability; no global platform factory mutation.

Call on the actual execution thread. Initialization/drain synchronize only at
explicit boundaries. Warmup holds its allocation until both streams finish and
never suppresses allocator errors. No model, weights, arithmetic or device reset.
"""
from __future__ import annotations
from contextlib import contextmanager
import os
import threading


class StreamOwnershipError(RuntimeError):
    pass


class OwnedCopyStream128:
    def __init__(self, torch, *, device: int):
        self.torch=torch;self.device=device
        self.pid=os.getpid();self.thread_id=threading.get_native_id()
        self.closed=False;self.ready=False;self._probe=None;self._host=None
        if torch.npu.current_device()!=device:
            raise StreamOwnershipError('Set the intended NPU on the execution thread before creation')
        self.compute=torch.npu.current_stream()
        self.copy=torch.npu.Stream(device=device)
        if self.copy==self.compute:
            raise StreamOwnershipError('Explicit private stream is not distinct; refuse hidden serial fallback')
        # Strict allocator registration on this same thread; keep both storages alive.
        self._host=torch.arange(8,dtype=torch.float16).pin_memory()
        self.copy.wait_stream(self.compute)
        with torch.npu.stream(self.copy):
            self._probe=torch.empty(8,dtype=torch.float16,device=f'npu:{device}')
            self._probe.copy_(self._host,non_blocking=True)
            ready=torch.npu.Event();ready.record(self.copy)
        self.compute.wait_event(ready)
        self.copy.synchronize();self.compute.synchronize()
        actual=self._probe.cpu().contiguous()
        if not torch.equal(actual.view(torch.uint8),self._host.view(torch.uint8)):
            raise RuntimeError('Strict owner-thread allocator warmup H2D bytes changed')
        self.ready=True

    def require_owner(self):
        # Check PID/thread BEFORE touching the runtime from an unregistered thread.
        if os.getpid()!=self.pid or threading.get_native_id()!=self.thread_id:
            raise StreamOwnershipError('Private stream capability cannot move between threads or processes')
        if self.closed or not self.ready:
            raise StreamOwnershipError('Private stream is not ready or already closed')
        if self.torch.npu.current_device()!=self.device:
            raise StreamOwnershipError('Private stream device changed')
        if self.torch.npu.current_stream()!=self.compute:
            raise StreamOwnershipError('Unexpected compute stream; use a separate explicitly-owned capability')

    @contextmanager
    def copying(self):
        self.require_owner()
        with self.torch.npu.stream(self.copy):
            yield self.copy
        self.require_owner()

    def drain(self):
        self.require_owner()
        self.copy.synchronize();self.compute.synchronize()

    def close(self):
        if self.closed:
            if os.getpid()!=self.pid or threading.get_native_id()!=self.thread_id:
                raise StreamOwnershipError('Closed capability still cannot move between threads')
            return
        self.drain()
        # References are not released until every use on both streams is complete.
        self._probe=None;self._host=None;self.closed=True

    def facts(self):
        self.require_owner()
        return {'pid':self.pid,'native_thread_id':self.thread_id,'device':self.device,
            'compute_stream':int(self.compute.npu_stream),'copy_stream':int(self.copy.npu_stream),
            'distinct_streams':self.copy!=self.compute,'strict_allocator_warmup_completed':self.ready,
            'global_platform_Stream_unchanged_by_this_capability':True,
            'warmup_probe_retained_until_owner_close':self._probe is not None}
