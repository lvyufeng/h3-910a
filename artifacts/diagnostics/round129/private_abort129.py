"""Bounded cleanup of EXACT experiment processes including detached worker children.

Environment capability + UID + /proc start ticks prevent touching other users or
PID-reused processes. No driver/device reset or shared process-group signalling.
"""
from pathlib import Path
import json
import os
import select
import signal
import time

ROOT=Path('/tmp/h3_repeat12902')
MARKER=b'H3_R127_IPC_KEY_FILE=/tmp/h3_repeat12902/private_IPC_key127.bin\0'


def private_rows129():
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


def verify_row129(row):
    proc=Path('/proc',str(row['pid']))
    try:
        return MARKER in (proc/'environ').read_bytes() and (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==row['start_ticks']
    except (FileNotFoundError,ProcessLookupError):return False


def abort_private129():
    rows=private_rows129();fds={}
    for row in rows:
        if not verify_row129(row):continue
        try:
            fd=os.pidfd_open(row['pid']);fds[fd]=row;os.kill(row['pid'],signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.monotonic()+30
    while fds and time.monotonic()<deadline:
        ready,_,_=select.select(list(fds),[],[],max(0,deadline-time.monotonic()))
        for fd in ready:os.close(fd);fds.pop(fd,None)
    killed=[]
    for fd,row in fds.items():
        if verify_row129(row):
            os.kill(row['pid'],signal.SIGKILL);killed.append(row['pid'])
    teardown_deadline=time.monotonic()+300
    for fd,row in fds.items():
        ready,_,_=select.select([fd],[],[],max(0,teardown_deadline-time.monotonic()));os.close(fd)
        assert ready,('Private cohort cleanup did not complete; no device reset allowed',row)
    assert not private_rows129(),'New private child remained; preserve unknown state, never claim cleanup'
    result={'private_only_verified_processes_signalled':rows,'after_grace_private_kill_pids':killed,
        'all_private_terminated':True,'no_shared_processes_or_device_state_changed':True}
    target=ROOT/'private_abort_details129.json'
    assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    return result
