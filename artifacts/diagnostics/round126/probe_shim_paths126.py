#!/usr/bin/env python3
"""Locate small runtime shim sources without importing Torch or touching devices."""
import hashlib
import importlib.util
import json
from pathlib import Path

root=Path('/tmp/h3_perf12601')
names=['h3_910a_sdpa_shim']
paths=[]
for name in names:
    spec=importlib.util.find_spec(name)
    if spec and spec.origin:paths.append(Path(spec.origin).resolve())
paths.extend(Path(p) for p in [
    '/home/mseco/vllm-omni/vllm_omni/h3_910a_sdpa_shim.py',
    '/home/mseco/vllm-omni/h3_910a_sdpa_shim.py',
    '/home/mseco/h3-910a/runtime/h3_910a_sdpa_shim.py',
    '/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/h3_910a_sdpa_shim.py',
])
found=[]
for p in paths:
    if p.is_file() and p not in [Path(x['path']) for x in found]:
        data=p.read_bytes()
        assert len(data)<2*1024**2 and p.suffix=='.py'
        out=root/'native_sources'/f'sdpa_shim_candidate{len(found)}.py';assert not out.exists()
        out.write_bytes(data)
        found.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest(),'snapshot':str(out)})
np=Path('/tmp/h3_align12401/npu_half_nan_compat124.py')
data=np.read_bytes()
r={'scope':__doc__,'completed':True,'sdpa_shim_candidates':found,
   'nan_compat_path':str(np),'nan_compat_sha256':hashlib.sha256(data).hexdigest(),
   'device_accessed':False,'torch_imported':False}
(root/'shim_paths126.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2),flush=True)
