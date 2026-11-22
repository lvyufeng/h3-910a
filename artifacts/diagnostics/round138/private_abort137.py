"""Bounded cleanup of exact private processes, with PID-fd-bound signals.

Environment capability + UID + /proc start ticks identify candidates. A pidfd is
opened and the evidence rechecked before signalling; PID numbers are never used
for signals. No shared process groups, driver changes or device reset.
"""
from pathlib import Path
import json
import os
import select
import signal
import threading
import time

ROOT=Path('/tmp/h3_vae13801')
MARKER=b'H3_R127_IPC_KEY_FILE=/tmp/h3_vae13801/private_IPC_key127.bin\0'
_ABORT_LOCK=threading.RLock()


def private_rows137():
    result=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal() or int(proc.name)==os.getpid():continue
        try:
            if MARKER not in (proc/'environ').read_bytes():continue
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]=='Z':continue
            assert proc.stat().st_uid==os.getuid(),'Private capability unexpectedly held by another UID'
            result.append({'pid':int(proc.name),'start_ticks':stat[19],
                'command':(proc/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')})
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result


def verify_row137(row):
    proc=Path('/proc',str(row['pid']))
    try:
        return (proc.stat().st_uid==os.getuid() and MARKER in (proc/'environ').read_bytes()
            and (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks'])
    except (FileNotFoundError,PermissionError,ProcessLookupError):return False


def abort_private137():
    # Monitor and post-parent-exit reconciliation may both request cleanup.
    with _ABORT_LOCK:return _abort_private137()


def _abort_private137():
    assert hasattr(signal,'pidfd_send_signal'),'Do not fall back to PID-number signals'
    rows=private_rows137();fds={};signalled=[];killed=[]
    result={'private_candidates':rows,'no_PID_number_signalling':True,'no_shared_processes_or_device_state_changed':True}
    target=ROOT/f'private_abort_details137_{os.getpid()}_{time.monotonic_ns()}.json'
    assert target.parent.resolve()==ROOT.resolve() and not target.exists()
    try:
        for row in rows:
            fd=None
            try:
                fd=os.pidfd_open(row['pid'])
                if not verify_row137(row):os.close(fd);fd=None;continue
                signal.pidfd_send_signal(fd,signal.SIGTERM)
                fds[fd]=row;signalled.append(row);fd=None
            except ProcessLookupError:pass
            finally:
                if fd is not None:os.close(fd)
        deadline=time.monotonic()+30
        while fds and time.monotonic()<deadline:
            ready,_,_=select.select(list(fds),[],[],max(0,deadline-time.monotonic()))
            for fd in ready:os.close(fd);fds.pop(fd,None)
        for fd,row in fds.items():
            # Recheck private ownership/capability, then signal only the bound original task.
            if not verify_row137(row):continue
            try:signal.pidfd_send_signal(fd,signal.SIGKILL);killed.append(row['pid'])
            except ProcessLookupError:pass
        teardown_deadline=time.monotonic()+300
        for fd,row in list(fds.items()):
            ready,_,_=select.select([fd],[],[],max(0,teardown_deadline-time.monotonic()))
            os.close(fd);fds.pop(fd,None)
            assert ready,('Private cohort teardown incomplete; no device reset allowed',row)
        assert not private_rows137(),'A newly spawned private child remains; do not claim cleanup'
        result['all_private_terminated']=True
    except BaseException as exc:
        result['all_private_terminated']=False;result['failure']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        for fd in fds:os.close(fd)
        result['private_only_verified_processes_signalled']=signalled;result['after_grace_private_kill_pids']=killed
        target.write_text(json.dumps(result,indent=2)+'\n')
    return result
