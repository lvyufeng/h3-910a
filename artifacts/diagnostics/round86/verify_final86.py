#!/usr/bin/env python3
"""Read-only R86 final worker, production, CPU oracle and runtime reconciliation."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path('/tmp/h3_round86_late01')
expected=json.loads((ROOT/'expected_sources76.json').read_text())
r={'scope':__doc__,'production':{},'installed_api_sources':{},'remaining_workers':[]}
for category in('production','installed_api_sources'):
    for name,wanted in expected[category].items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==wanted,name
        r[category][name]={'sha256':actual,'unchanged':True}
for name,wanted in(('/tmp/h3_round78_hf_fp1601/reference.json','44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'),
                   ('/tmp/libfa910a.so','bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043')):
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==wanted,name
    r[name]={'sha256':actual,'unchanged':True}
commands=('late_rebase86.py','prebuild_late86.py')
for p in Path('/proc').iterdir():
    if not p.name.isdecimal():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if cmd.startswith('/home/mseco/vllm910a/bin/python')and any(name in cmd for name in commands):
            r['remaining_workers'].append({'pid':int(p.name),'command':cmd})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not r['remaining_workers']
r['all_production_reference_and_launcher_unchanged']=True
print(json.dumps(r,indent=2))
