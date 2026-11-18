#!/usr/bin/env python3
"""Read-only R89 originalHF/reference,production/runtime and worker cleanup check."""
from pathlib import Path
import json
import hashlib
import os
ROOT=Path('/tmp/h3_round89_vision02')
expected=json.loads((ROOT/'expected_sources76.json').read_text())
r={'scope':__doc__,'production':{},'installed_api_sources':{},'remaining_workers':[]}
for category in('production','installed_api_sources'):
    for name,wanted in expected[category].items():
        actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==wanted,name
        r[category][name]={'sha256':actual,'unchanged':True}
for name,wanted in(('/tmp/h3_round78_hf_fp1601/reference.json','44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'),
                   ('/tmp/h3_round78_hf_fp1601/FP16_vision.pt','c01bbe180f454fe44b182bd9307491ef0f4c342b6c2ff6be08d1e3453d8c312a'),
                   ('/tmp/h3_round78_hf_fp1601/FP16_selected50.pt','0137a434148366c3dcbf825970274ad0faa036666561d7aecc0170cf136b3fe4'),
                   ('/tmp/libfa910a.so','bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043')):
    actual=hashlib.sha256(Path(name).read_bytes()).hexdigest();assert actual==wanted,name
    r[name]={'sha256':actual,'unchanged':True}
reference=json.loads(Path('/tmp/h3_round78_hf_fp1601/reference.json').read_text())
hf=Path(reference['HF_modeling_source_path']);actual=hashlib.sha256(hf.read_bytes()).hexdigest()
assert actual==reference['HF_modeling_source_sha256']=='fcb5571ef95e4a6679af866ef9ef4dfa5cee017f9f91453daf28c7804c7a50fe'
r['original_HF_modeling_source']={'path':str(hf),'sha256':actual,'unchanged':True}
commands=('vision_probe89.py','vision_probe_fixed89.py')
for p in Path('/proc').iterdir():
    if not p.name.isdecimal():continue
    try:
        if p.stat().st_uid!=os.getuid():continue
        cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
        if cmd.startswith('/home/mseco/vllm910a/bin/python')and any(name in cmd for name in commands):
            r['remaining_workers'].append({'pid':int(p.name),'command':cmd})
    except(FileNotFoundError,PermissionError,ProcessLookupError):continue
assert not r['remaining_workers']
r['all_production_reference_originalHF_and_launcher_unchanged']=True
print(json.dumps(r,indent=2))
