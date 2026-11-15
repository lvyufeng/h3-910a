#!/usr/bin/env python3
"""Verify every already-written native tensor's raw bits against the online cast."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

root=Path('/tmp/h3_round53_validation01')
r=json.loads((root/'native_storage_conversion.json').read_text())
assert r['complete'] and len(r['files'])==13
src=Path(r['source']); dst=Path(r['destination'])
count=0
for item in r['files']:
    with safe_open(str(src/item['filename']),framework='pt',device='cpu') as a, safe_open(str(dst/item['filename']),framework='pt',device='cpu') as b:
        assert set(a.keys())==set(b.keys())
        for key in a.keys():
            original=a.get_tensor(key)
            expected=original.half() if original.dtype==torch.bfloat16 else original
            got=b.get_tensor(key)
            assert got.dtype==expected.dtype
            itype={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[got.element_size()]
            assert torch.equal(got.view(itype),expected.contiguous().view(itype)),key
            count+=1
out=root/'native_storage_bitcheck.json'
if out.exists(): raise RuntimeError('Refuse to overwrite a verification receipt')
report={'passed':True,'shards':13,'tensors_checked':count,'raw_bits_match_existing_runtime_cast':True,
        'fp32_values_preserved':True,'native_unfused_weights':True,
        'qualification':'Matching current FP16 runtime conversion including signed zeros; not BF16 inference or GPU-reference lossless proof'}
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
