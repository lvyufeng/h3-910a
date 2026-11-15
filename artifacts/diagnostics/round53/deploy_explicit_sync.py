#!/usr/bin/env python3
"""Replace the rejected conditional-sync timer with explicit PERF-only drain."""
import ast
import hashlib
import json
from pathlib import Path

root=Path('/tmp/h3_round53_validation01')
target=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py')
assert target.resolve(strict=True)==target
previous=json.loads((root/'deployment.json').read_text())
expected=next(x['after_sha256'] for x in previous['sources'] if x['target']==str(target))
old=target.read_bytes(); assert hashlib.sha256(old).hexdigest()==expected
new=(root/'candidate_denoise_loop.py').read_bytes(); ast.parse(new.decode())
assert b'current_omni_platform.synchronize()' in new
backup=Path(str(target)+'.pre_explicit_perf_sync_20261115.bak')
with backup.open('xb') as f:f.write(old)
target.write_bytes(new); assert target.read_bytes()==new
report={'target':str(target),'backup':str(backup),'before_sha256':expected,'after_sha256':hashlib.sha256(new).hexdigest(),
        'measurement_contract':'PERF=1 adds explicit current_omni_platform.synchronize before complete-step timer; successful cancellation checks are not drains; PERF=0 stays asynchronous'}
(root/'deployment_explicit_sync.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
