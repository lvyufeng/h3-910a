#!/usr/bin/env python3
"""Deploy validated default-off vision precision mode, retaining source backup."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_round55_vision01')
target=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py')
assert target.resolve(strict=True)==target
old=target.read_bytes();assert old==(ROOT/'source/encoder.py').read_bytes()
validation=json.loads((ROOT/'optin_validation.json').read_text());assert validation['completed'] and validation['default_disabled_npu_linear_and_interpolation_exact']
new=(ROOT/'candidate_encoder.py').read_bytes();ast.parse(new.decode())
assert b'get("VLLM_OMNI_910A_VISION_FP32", "0")' in new
backup=Path(str(target)+'.pre_vision_fp32_optin_20261116.bak')
with backup.open('xb') as f:f.write(old)
target.write_bytes(new);assert target.read_bytes()==new
r={'target':str(target),'backup':str(backup),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest(),
   'default_enabled':False,'optin':'VLLM_OMNI_910A_VISION_FP32=1','stored_parameter_dtype':'BF16 unchanged','output_dtype':'BF16 unchanged',
   'global_hf32_flags_modified':False,'scope':'Optional higher-precision vision linear/interpolation compute; full generation/reference quality still requires validation'}
(ROOT/'deployment.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
