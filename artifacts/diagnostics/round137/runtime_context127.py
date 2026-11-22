"""Request-scoped observational context; no torch import, model cache or arithmetic."""
from __future__ import annotations
import json
import os
from pathlib import Path
import threading
import time

ROOT=Path('/tmp/h3_components13701')
SESSION=Path(os.environ['H3_R127_SESSION_OUT']).resolve()
assert SESSION.parent==ROOT and SESSION.name=='engine_components137a' and SESSION.is_dir()
LABELS={1:'r01_cold',2:'r02_warm',3:'r03_native_reload'}
_CURRENT=0
_SEQUENCE=0
_BUFFER=[]
_LOCK=threading.Lock()


def request_index():return _CURRENT


def request_directory():
    assert _CURRENT in LABELS,('No active native request',_CURRENT)
    path=SESSION/LABELS[_CURRENT]
    assert path.is_dir()
    return path


def event(name,**data):
    global _SEQUENCE
    with _LOCK:
        _SEQUENCE+=1
        _BUFFER.append({'event':name,'pid':os.getpid(),'rank':int(os.environ.get('LOCAL_RANK',os.environ.get('RANK','-1'))),
            'thread_id':threading.get_native_id(),'unix':time.time(),'sequence':_SEQUENCE,'request_index':_CURRENT,**data})


def flush():
    with _LOCK:
        rows=list(_BUFFER);_BUFFER.clear()
    if rows:
        with (SESSION/f'worker_runtime_pid{os.getpid()}127.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def begin_request(index):
    global _CURRENT
    assert _CURRENT==0 and index in LABELS
    _CURRENT=index
    event('request_context_begin',label=LABELS[index])


def end_request(*,success):
    global _CURRENT
    assert _CURRENT in LABELS
    index=_CURRENT;directory=request_directory()
    event('request_context_end',label=LABELS[index],success=success)
    flush();_CURRENT=0
    from audit_completion127 import publish_complete
    publish_complete(directory,index,os.getpid(),success=success)
