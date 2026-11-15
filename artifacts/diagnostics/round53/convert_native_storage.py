#!/usr/bin/env python3
"""Unfused native storage conversion matching the existing FP16 runtime cast.

Not a claim of bit-exact BF16 inference. BF16->FP16->BF16 differences are counted.
FP32 tensors remain FP32. Every saved tensor must match the online cast exactly.
No model/LoRA fusion, source edits, move/unlink, or overwriting is performed.
"""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE=Path('/home/mseco/models/MiniMax-H3/FL2VA')
DEST=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53')
ALIAS=Path('/tmp/h3_round53_native49_fp16_model')
REPORT=Path('/tmp/h3_round53_validation01/native_storage_conversion.json')
for p in (DEST,ALIAS,REPORT):
    if p.exists(): raise RuntimeError('Refuse existing conversion artifact: '+str(p))
assert SOURCE.resolve(strict=True)==SOURCE
DEST.mkdir(); weights=DEST/'FL2VA/transformer'; weights.mkdir(parents=True)
receipt={'source':str(SOURCE/'transformer'),'destination':str(weights),'files':[],
         'precision_contract':'Storage matches BF16-to-FP16 cast of existing --dtype half runtime; F32 remains unchanged; not BF16 lossless proof'}
for i in range(1,14):
    name=f'model-{i:05d}-of-00013.safetensors'
    src=SOURCE/'transformer'/name; dst=weights/name
    tensors={}; changed=0; dtypes={}
    with safe_open(str(src),framework='pt',device='cpu') as reader:
        for key in reader.keys():
            tensor=reader.get_tensor(key)
            original_dtype=str(tensor.dtype)
            result=tensor.to(torch.float16) if tensor.dtype==torch.bfloat16 else tensor
            if tensor.dtype==torch.bfloat16:
                changed+=int(torch.count_nonzero(result.to(torch.bfloat16)!=tensor))
                if not bool(torch.isfinite(result).all()): raise RuntimeError('Nonfinite cast at '+key)
            tensors[key]=result.contiguous()
            dtypes[original_dtype]=dtypes.get(original_dtype,0)+1
        save_file(tensors,str(dst),metadata={'format':'pt'})
    with safe_open(str(dst),framework='pt',device='cpu') as check:
        assert set(check.keys())==set(tensors)
        for key,value in tensors.items():
            saved=check.get_tensor(key)
            assert saved.dtype==value.dtype and torch.equal(saved,value),'Saved runtime cast mismatch: '+key
            integer_dtype={1:torch.int8,2:torch.int16,4:torch.int32,8:torch.int64}[saved.element_size()]
            assert torch.equal(saved.view(integer_dtype),value.view(integer_dtype)), 'Saved bit-pattern mismatch: '+key
    digest=hashlib.sha256()
    with dst.open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):digest.update(chunk)
    item={'filename':name,'bytes':dst.stat().st_size,'sha256':digest.hexdigest(),'tensors':len(tensors),
          'source_dtype_tensor_counts':dtypes,'bf16_roundtrip_changed_elements':changed,
          'all_saved_tensors_equal_online_cast':True}
    receipt['files'].append(item)
    REPORT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(item),flush=True)
    del tensors,tensor,result,value,saved
# Native index/config can be reused because shard filenames/keys are unchanged.
for name in ('config.json','model.safetensors.index.json'):
    source=SOURCE/'transformer'/name
    assert source.exists()
    (weights/name).symlink_to(source.resolve(strict=True))
ALIAS.mkdir(); partition=ALIAS/'FL2VA'; partition.mkdir()
metadata=Path('/tmp/h3_round53_native49_model/FL2VA/model_index.json').read_bytes()
parsed=json.loads(metadata); assert 'base_schedule' not in parsed['_minimax_h3']
for entry in SOURCE.iterdir():
    if entry.name=='model_index.json': continue
    target=weights if entry.name=='transformer' else entry.resolve(strict=True)
    (partition/entry.name).symlink_to(target,target_is_directory=entry.is_dir())
(partition/'model_index.json').write_bytes(metadata)
(ALIAS/'model_index.json').write_bytes(metadata)
receipt.update({'alias':str(ALIAS),'complete':True,'total_bytes':sum(x['bytes'] for x in receipt['files']),
                'total_bf16_roundtrip_changed_elements':sum(x['bf16_roundtrip_changed_elements'] for x in receipt['files']),
                'all_saved_tensors_equal_online_cast':True})
REPORT.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2),flush=True)
