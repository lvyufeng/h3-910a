#!/usr/bin/env python3
"""Bounded current TP8 state for an existing run;never creates/replays a model job."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_round84_row_fp3201')
r={'root':str(ROOT),'run_started':None,'run_finished':None,'rank_receipts':[],'matching_processes':[]}
for name in('run_started.json','run_finished.json'):
    p=ROOT/name
    if p.exists():r[name[:-5]]=json.loads(p.read_text())
for rank in range(8):
    p=ROOT/('tp8_rank%d.json'%rank)
    if not p.exists():continue
    data=p.read_bytes();assert len(data)<4*1024*1024;record=json.loads(data)
    r['rank_receipts'].append({'rank':rank,'sha256':hashlib.sha256(data).hexdigest(),'completed':record.get('completed'),
                              'failure':record.get('failure'),'loaded_parameter_tensors':record.get('loaded_parameter_tensors'),
                              'completed_language_layers':len(record.get('language_layers',[])),'case_keys':list(record.get('cases',{}))})
for p in Path('/proc').iterdir():
    if not p.name.isdecimal():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if '/tmp/h3_round80_full_fp16_01/full_row_fp3284.py'in cmd:
            r['matching_processes'].append({'pid':int(p.name),'command':cmd[:512]})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(r,indent=2))
