#!/usr/bin/env python3
"""Deploy only audited timing/collective edits; retain source and model metadata."""
import ast
import hashlib
import json
from pathlib import Path

root=Path('/tmp/h3_round53_validation01')
base=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3')
receipts=[]
for name,suffix in (('encoder.py','pre_reduce_return_20261115'),('denoise_loop.py','pre_complete_timer_20261115')):
    target=base/name
    assert target.resolve(strict=True)==target
    old=target.read_bytes()
    inspected=(root/'source'/name).read_bytes()
    assert old==inspected,'Source changed since inspection: '+name
    candidate=(root/('candidate_'+name)).read_bytes()
    ast.parse(candidate.decode())
    backup=Path(str(target)+'.'+suffix+'.bak')
    with backup.open('xb') as f:f.write(old)
    target.write_bytes(candidate)
    assert target.read_bytes()==candidate
    receipts.append({'target':str(target),'backup':str(backup),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(candidate).hexdigest()})
# Root dispatch metadata points to the same native partition, not the old
# externally fused student's root class or pinned four-interval schedule.
model=Path('/tmp/h3_round53_native49_model')
meta=(model/'FL2VA/model_index.json').read_bytes()
parsed=json.loads(meta)
assert 'base_schedule' not in parsed['_minimax_h3']
assert parsed['_class_name']=='MiniMaxH3Pipeline'
with (model/'model_index.json').open('xb') as f:f.write(meta)
report={'sources':receipts,'private_root_metadata_sha256':hashlib.sha256(meta).hexdigest(),
        'scope':'Collective return handling and additive synchronized timing; original unfused model alias, originals unchanged'}
(root/'deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
