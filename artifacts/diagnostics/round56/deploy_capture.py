#!/usr/bin/env python3
"""Install math-preserving focused text captures with exact source backup."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_round56_audit01');target=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py')
assert target.resolve(strict=True)==target
old=target.read_bytes();assert old==(ROOT/'source/encoder.py').read_bytes()
assert hashlib.sha256(old).hexdigest()=='f04123afdeda7cceb9f98e84f3873644fd64d0c37e2e24d4490959a14b90ab32'
new=(ROOT/'candidate_capture_encoder.py').read_bytes();ast.parse(new.decode())
helper=target.parent/'h3_text_audit56.py'
if helper.exists():raise RuntimeError('Refuse existing audit helper overwrite')
with helper.open('xb') as f:f.write((ROOT/'h3_text_audit56.py').read_bytes())
backup=Path(str(target)+'.pre_round56_capture_20261116.bak')
with backup.open('xb') as f:f.write(old)
target.write_bytes(new);assert target.read_bytes()==new
r={'target':str(target),'backup':str(backup),'helper':str(helper),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest(),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'scope':'Temporary late-text data/context capture, no numerical changes'}
(ROOT/'capture_deployment.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
