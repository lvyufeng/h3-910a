"""Out-of-band cohort setup/status and fail-closed private-launcher monitor.

TCP Store phases do not depend on a working NPU collective. An unrecoverable
rank-local exception publishes a private atomic failure receipt immediately;
the launcher aborts only its verified experiment process group, not shared jobs.
Numerical differences are ordinary false gates, not exceptions or tolerance.
"""
from __future__ import annotations
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import select
import signal
import struct
import threading

ROOT=Path('/tmp/h3_full5013101')
FAILURE=ROOT/'cohort_failure128.json'


def publish_fatal128(row):
    # No NPU access: usable even when a rank failed before a TCP Store capability exists.
    target=FAILURE;assert target.resolve()==Path('/tmp/h3_full5013101/cohort_failure128.json')
    temporary=ROOT/f'cohort_failure_pid{os.getpid()}_{__import__("time").monotonic_ns()}.json'
    assert not temporary.exists();temporary.write_text(json.dumps(row,indent=2)+'\n')
    try:os.link(temporary,target)
    except FileExistsError:pass  # Preserve first failure and the new per-rank receipt separately.


class CohortPhaseError(RuntimeError):
    pass


class Cohort128:
    def __init__(self,torch,*,fatal_destination=FAILURE,prefix='R128formal',timeout_s=120):
        self.rank=torch.distributed.get_rank();self.world=torch.distributed.get_world_size()
        assert self.world==8
        self.store=torch.distributed.distributed_c10d._get_default_store()
        self.prefix=prefix;self.sequence=0;self.timeout=timedelta(seconds=timeout_s)
        self.fatal_destination=fatal_destination

    def _publish_fatal(self,row):
        if self.fatal_destination is None:return
        target=self.fatal_destination
        assert target.resolve()==FAILURE and target.parent==ROOT
        publish_fatal128(row)

    def run(self,label,function):
        index=self.sequence;self.sequence+=1
        key=f'{self.prefix}/{index}:{label}/rank{self.rank}'
        value=None;error=None
        try:value=function();row={'ok':True,'phase':label,'rank':self.rank,'pid':os.getpid()}
        except BaseException as exc:
            error=exc;row={'ok':False,'phase':label,'rank':self.rank,'pid':os.getpid(),'type':type(exc).__name__,'message':str(exc)}
            self._publish_fatal(row)
        self.store.set(key,json.dumps(row).encode())
        keys=[f'{self.prefix}/{index}:{label}/rank{rank}' for rank in range(self.world)]
        try:self.store.wait(keys,self.timeout)
        except BaseException as exc:
            self._publish_fatal({'ok':False,'phase':label,'rank':self.rank,'pid':os.getpid(),'type':type(exc).__name__,'message':'Cohort status timeout/failure: '+str(exc)})
            raise CohortPhaseError('Cohort phase did not reach all ranks: '+label) from exc
        rows=[json.loads(self.store.get(k).decode()) for k in keys]
        failures=[r for r in rows if not r['ok']]
        if failures:raise CohortPhaseError(json.dumps({'phase':label,'failures':failures})) from error
        return value


def prepare_failure_watch128():
    # Validate monitor capability synchronously BEFORE spawning any model process.
    libc=ctypes.CDLL(None,use_errno=True)
    libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_init1.restype=ctypes.c_int
    libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32];libc.inotify_add_watch.restype=ctypes.c_int
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1 for private failure monitor')
    watch=libc.inotify_add_watch(fd,str(ROOT).encode(),0x100|0x80|0x8)
    if watch<0:
        error=ctypes.get_errno();os.close(fd);raise OSError(error,'inotify_add_watch for private failure monitor')
    assert not FAILURE.exists(),'An earlier failed cohort must be preserved, not replayed'
    return fd


def monitor_private_failure(process,stop_event,fd,*,deadline_monotonic=None):
    """Total monotonic deadline and atomic-fatal monitor; no reset on file events."""
    import time
    if deadline_monotonic is None:deadline_monotonic=time.monotonic()+5400
    def abort(reason,row=None):
        process._private_abort_called131=True
        from private_abort131 import abort_private131
        print(json.dumps({'private_cohort_abort_requested':process.pid,'reason':reason,'failure':row,'no_shared_processes_signalled':True}),flush=True)
        abort_private131()
    wake_read,wake_write=os.pipe()
    def wake_on_finish():stop_event.wait();os.write(wake_write,b'1')
    wake=threading.Thread(target=wake_on_finish,daemon=True);wake.start()
    try:
        while not stop_event.is_set():
            remaining=deadline_monotonic-time.monotonic()
            if remaining<=0:
                abort('total_90minute_compute_budget_exceeded');return
            if FAILURE.exists():
                # Parent exit is not evidence that detached private workers exited.
                abort('atomic_cohort_failure',json.loads(FAILURE.read_text()));return
            readable,_,_=select.select([fd,wake_read],[],[],remaining)
            if wake_read in readable:return
            if fd in readable:os.read(fd,65536)
            # A select timeout re-enters the total deadline branch, never a fresh interval.
    finally:
        os.close(fd)
        stop_event.set();wake.join(timeout=5)
        os.close(wake_read);os.close(wake_write)
