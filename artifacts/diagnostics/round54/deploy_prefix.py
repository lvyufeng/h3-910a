#!/usr/bin/env python3
"""Deploy verified exact-prefix attention, restore encoder capture sites."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_round54_validation01')
previous=json.loads((ROOT/'deployment.json').read_text())
receipts=[]
for item in previous['sources']:
    target=Path(item['target']);assert target.resolve(strict=True)==target
    current=target.read_bytes();assert hashlib.sha256(current).hexdigest()==item['after_sha256']
    if target.name=='encoder.py':
        new=(ROOT/'source/encoder.py').read_bytes()
        assert hashlib.sha256(new).hexdigest()==item['before_sha256']
        change='restore diagnostic encoder capture sites; runtime NPU patches untouched'
    else:
        new=(ROOT/'candidate_prefix_fa910a.py').read_bytes()
        assert b'_k%d_v3' in new and b'corr = 1.0' not in new
        change='exact valid-key prefix masking including arbitrary sub-16 tail; remove invalid zero-key LSE assumption'
    ast.parse(new.decode())
    backup=Path(str(target)+'.pre_exact_prefix_20261116.bak')
    with backup.open('xb') as f:f.write(current)
    target.write_bytes(new);assert target.read_bytes()==new
    receipts.append({'target':str(target),'backup':str(backup),'before_sha256':item['after_sha256'],
                     'after_sha256':hashlib.sha256(new).hexdigest(),'change':change})
report={'sources':receipts,'capture_helper_retained_but_unreferenced':previous['helper'],
        'qualification':'Prefix correctness patch validated on actual tensors; whole-model quality must be rechecked'}
(ROOT/'prefix_deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
