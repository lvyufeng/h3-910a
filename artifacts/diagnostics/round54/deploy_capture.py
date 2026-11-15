#!/usr/bin/env python3
"""Install bounded opt-in diagnostics; do not overwrite any earlier source."""
import ast
import hashlib
import json
from pathlib import Path

root=Path('/tmp/h3_round54_validation01')
backends=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends')
encoder=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py')
helper=backends/'h3_910a_audit.py'
assert not helper.exists()
with helper.open('xb') as f:f.write((root/'h3_910a_audit.py').read_bytes())
receipt={'helper':str(helper),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'sources':[]}
for target in (encoder,backends/'fa910a.py'):
    assert target.resolve(strict=True)==target
    old=target.read_bytes(); assert old==(root/'source'/target.name).read_bytes()
    new=(root/('candidate_'+target.name)).read_bytes(); ast.parse(new.decode())
    backup=Path(str(target)+'.pre_audit_capture_20261116.bak')
    with backup.open('xb') as f:f.write(old)
    target.write_bytes(new); assert target.read_bytes()==new
    receipt['sources'].append({'target':str(target),'backup':str(backup),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest()})
(root/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
