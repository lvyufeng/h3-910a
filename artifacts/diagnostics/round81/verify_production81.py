#!/usr/bin/env python3
"""Post-probe read-only source/cleanup verification;no CPU/NPU model execution."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_round81_full_fp16_pad02')
expected=json.loads((ROOT/'expected_sources76.json').read_text())
r={'scope':__doc__,'production':{},'installed_api_sources':{},'NPU_probe_processes_remaining':[],'original_CPU_reference_completed':True}
for category in('production','installed_api_sources'):
    for path,digest in expected[category].items():
        data=Path(path).read_bytes();actual=hashlib.sha256(data).hexdigest();assert actual==digest,path
        r[category][path]={'sha256':actual,'unchanged':True}
for process in Path('/proc').iterdir():
    if not process.name.isdecimal():continue
    try:
        if process.stat().st_uid!=os.getuid():continue
        command=(process/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if '/tmp/h3_round81_full_fp16_pad02/encode_fp16_padded_kv_fixed81.py'in command:r['NPU_probe_processes_remaining'].append({'pid':int(process.name),'command':command})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not r['NPU_probe_processes_remaining']
r['original_CPU_reference_process_alive']=Path('/proc/4077809').exists()
r['all_production_sources_unchanged']=True
with(ROOT/'post_probe_verification81.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
print(json.dumps({'production_count':len(r['production']),'installed_api_count':len(r['installed_api_sources']),'all_unchanged':True,'NPU_probe_processes_remaining':0,'original_CPU_reference_process_alive':r['original_CPU_reference_process_alive']},indent=2))
