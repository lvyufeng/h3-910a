"""Publish/read only request-audit completion, separate from model execution.

Markers are atomically linked after the JSONL file closes. The reader waits with
inotify for matching events under one already-authorized private directory; no
model launch, device collective, busy polling or sleep is used.
"""
from __future__ import annotations
import ctypes
import errno
import json
import os
from pathlib import Path
import select
import time
import uuid


def publish_complete(directory:Path,index:int,pid:int,*,success:bool):
    directory=directory.resolve();assert directory.is_dir() and index in (1,2,3) and pid>0
    target=directory/f'audit_complete_pid{pid}127.json'
    assert not target.exists(),'Completion must be published exactly once'
    temporary=directory/f'.audit_complete_pid{pid}.{uuid.uuid4().hex}.tmp'
    payload={'request_index':index,'pid':pid,'success':success,'audit_jsonl_closed':True}
    try:
        with temporary.open('x') as f:json.dump(payload,f);f.write('\n')
        os.link(temporary,target) # Atomic, no overwrite of an existing receipt.
    finally:
        if temporary.exists():
            assert temporary.resolve().parent==directory and temporary.name.startswith(f'.audit_complete_pid{pid}.')
            temporary.unlink()


def read_complete(directory:Path,index:int,expected=8):
    rows=[]
    for path in directory.glob('audit_complete_pid*127.json'):
        row=json.loads(path.read_text());assert row['request_index']==index and row['audit_jsonl_closed']
        rows.append(row)
    pids={r['pid'] for r in rows};assert len(pids)==len(rows)<=expected,'Invalid or duplicate completion markers'
    if any(not r['success'] for r in rows):raise RuntimeError('A worker failed; preserve the actual audit and do not start next request')
    return rows if len(rows)==expected else None


def wait_complete(directory:Path,index:int,*,expected=8,timeout_s=120):
    start=time.monotonic()
    ready=read_complete(directory,index,expected)
    if ready is not None:return {'pids':sorted(r['pid'] for r in ready),'audit_wait_s':time.monotonic()-start}
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        # IN_CREATE | IN_MOVED_TO | IN_CLOSE_WRITE; link() of the final marker is IN_CREATE.
        watch=libc.inotify_add_watch(fd,os.fsencode(directory),0x100|0x80|0x8)
        if watch<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while True:
            ready=read_complete(directory,index,expected) # Also closes the subscribe/read race.
            if ready is not None:return {'pids':sorted(r['pid'] for r in ready),'audit_wait_s':time.monotonic()-start}
            remaining=timeout_s-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('Missing worker audit completion; no next request is allowed')
            readable,_,_=select.select([fd],[],[],remaining)
            if not readable:raise TimeoutError('Missing worker audit completion; no next request is allowed')
            try:os.read(fd,65536)
            except BlockingIOError:pass
    finally:os.close(fd)
