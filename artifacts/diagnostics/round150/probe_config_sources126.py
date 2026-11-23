#!/usr/bin/env python3
"""Snapshot tiny component configuration and runtime shim files; no weights."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_fa150')
checkpoint=Path('/home/mseco/models/MiniMax-H3/FL2VA')
r={'scope':__doc__,'completed':True,'config_files':{},'source_files':{}}
for part in ('video_vae','audio_vae'):
    for path in sorted((checkpoint/part).glob('**/config*.json')):
        data=path.read_bytes();assert len(data)<1024**2
        r['config_files'][str(path)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
for path,label in [(Path('/home/mseco/vllm-omni/vllm_omni/h3_910a_sdpa_shim.py'),'sdpa_shim'),
                   (Path('/tmp/h3_align12401/npu_half_nan_compat124.py'),'nan_compat')]:
    data=path.read_bytes();r['source_files'][str(path)]={'import_name':'vllm_omni.h3_910a_sdpa_shim' if label=='sdpa_shim' else 'npu_half_nan_compat124','path':str(path),'sha256':hashlib.sha256(data).hexdigest()}
    out=ROOT/'native_sources'/f'{label}.py'
    if out.exists():assert out.read_bytes()==data
    else:out.write_bytes(data)
(ROOT/'config_sources126.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)
