#!/usr/bin/env python3
"""Read-only final production,CPU reference and exact-worker verification."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/tmp/h3_round82_condition_isolation01')
expected=json.loads((ROOT/'expected_sources76.json').read_text())
r={'scope':__doc__,'production':{},'installed_api_sources':{},'remaining_workers':[]}
for category in('production','installed_api_sources'):
    for path,wanted in expected[category].items():
        actual=hashlib.sha256(Path(path).read_bytes()).hexdigest();assert actual==wanted,path
        r[category][path]={'sha256':actual,'unchanged':True}
reference=Path('/tmp/h3_round78_hf_fp1601/reference.json')
r['CPU_reference_receipt_sha256']=hashlib.sha256(reference.read_bytes()).hexdigest()
assert r['CPU_reference_receipt_sha256']=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
names=('condition_isolation82.py','run_gemm_probe82.py','run_tik_gemm_probe82.py','run_tik_legacy_probe82.py','hf_cpu_fp16_reference78.py')
for entry in Path('/proc').iterdir():
    if not entry.name.isdecimal():continue
    try:
        if entry.stat().st_uid!=os.getuid():continue
        command=(entry/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if command.startswith('/home/mseco/vllm910a/bin/python')and any(name in command for name in names):
            r['remaining_workers'].append({'pid':int(entry.name),'command':command})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not r['remaining_workers']
r['all_production_unchanged']=True
print(json.dumps(r,indent=2))
