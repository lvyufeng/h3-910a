#!/usr/bin/env python3
"""Create a private native-FL2VA alias; never modify original metadata/weights."""
import hashlib
import json
from pathlib import Path
from safetensors import safe_open

ROOT=Path('/tmp/h3_round53_native49_model')
SOURCE=Path('/home/mseco/models/MiniMax-H3/FL2VA')
if ROOT.exists(): raise RuntimeError('Refuse to overwrite a model alias')
index_path=SOURCE/'model_index.json'
original=index_path.read_bytes()
index=json.loads(original)
assert index['_minimax_h3']['partition']=='fl2va'
assert 'fl2va' in index['_minimax_h3']['tasks']
removed=index['_minimax_h3'].pop('base_schedule',None)
ROOT.mkdir(); target=ROOT/'FL2VA'; target.mkdir()
for entry in SOURCE.iterdir():
    if entry.name=='model_index.json': continue
    (target/entry.name).symlink_to(entry.resolve(strict=True),target_is_directory=entry.is_dir())
(target/'model_index.json').write_text(json.dumps(index,indent=2)+'\n')
assert index_path.read_bytes()==original
# Binary metadata/small slices only: no all-weight copies, fusion or reconversion.
orig_file=SOURCE/'transformer/model-00001-of-00013.safetensors'
fused_file=Path('/home/mseco/models/MiniMax-H3-fp16/FL2VA/transformer/model-00001-of-00013.safetensors')
samples=[]
with safe_open(str(orig_file),framework='pt',device='cpu') as orig, safe_open(str(fused_file),framework='pt',device='cpu') as fused:
    shared=set(fused.keys())
    for key in orig.keys():
        view=orig.get_slice(key)
        shape=view.get_shape()
        if len(shape)!=2 or key not in shared: continue
        a=view[:4,:4].half().float(); b=fused.get_slice(key)[:4,:4].float()
        samples.append({'key':key,'shape':shape,'original_dtype':view.get_dtype(),
                        'fused_fp16_dtype':fused.get_slice(key).get_dtype(),
                        'sample_difference_after_matching_fp16_max':float((a-b).abs().max())})
        if len(samples)==8: break
report={'root':str(ROOT),'partition':str(target),'source':str(SOURCE),
        'original_index_sha256_unchanged':hashlib.sha256(original).hexdigest(),
        'removed_locally_pinned_schedule_from_alias':removed,
        'new_schedule':'No pinned distilled schedule; request explicitly asks 49 native intervals',
        'alias_uses_original_unfused_transformer':str((target/'transformer').resolve()),
        'small_binary_weight_samples':samples,
        'qualification':'Task-compatible original weights with existing native backend; not by itself official lossless or model-quality proof'}
Path('/tmp/h3_round53_validation01/native_model_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
