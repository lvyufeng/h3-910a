#!/usr/bin/env python3
"""Verify used CPU checkpoint shards and archive an atomic reference snapshot.

Reads only source metadata/receipts,not model tensor payloads;does not replay the
CPU reference. A verified first-pass vision stage is still not a complete
repeat-verified encoder oracle or NPU precision acceptance.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_round79_fp16_vision01')
CPU=Path('/tmp/h3_round78_hf_fp1601')
r=json.loads((CPU/'reference.json').read_text())
manifest=json.loads((ROOT/'official_text_manifest_receipt78.json').read_text())
verified={}
for name,info in r['checkpoint_shards'].items():
    assert info==manifest['lfs_shards'][name],name
    verified[name]=info
result={'scope':__doc__,'original_CPU_source_sha256':r['source_sha256'],'original_selected_parameter_sha256':r['original_selected_parameter_sha256'],
        'official_revision':manifest['revision'],'used_checkpoint_shards_verified_against_official_LFS':verified,
        'completed_vision_layers':len(r.get('vision_layers',[])),'completed_language_layers':len(r.get('language_layers',[])),
        'CPU_full_reference_completed':r['full_multimodal_reference_completed'],'CPU_repeat_pre_norm_equal':r.get('repeat_pre_final_norm_bits_equal'),
        'CPU_first_vision_tensors':r.get('stages',{}).get('FP16_vision'),'CPU_reference_failure':r.get('failure'),
        'NPU_precision_accepted':False,'whole_goal_complete':False}
path=ROOT/'CPU_source_verification79.json'
with path.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
