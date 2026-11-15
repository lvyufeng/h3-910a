#!/usr/bin/env python3
"""Remove capture sites and install only validated default-off text math mode.

The faster text-GEMM prototype is deliberately not deployed: its combination
with HF norm/RoPE worsens whole-encoder oracle error despite isolated gains.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/h3_round56_audit01');target=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/encoder.py')
assert target.resolve(strict=True)==target
prior=json.loads((ROOT/'capture_deployment.json').read_text());old=target.read_bytes();assert hashlib.sha256(old).hexdigest()==prior['after_sha256']
validation=json.loads((ROOT/'text_math_validation.json').read_text());assert validation['completed']
assert all(x['disabled_exactly_original'] for x in validation['norm_cases'])
assert all(y['raw_bits_equal'] for x in validation['rope_cases'] for y in x['comparisons'])
for rank in range(8):
 r=json.loads((ROOT/f'embedding_rank{rank}.json').read_text());assert r['completed'] and r['disabled_exact'] and all(x['new']['raw_bits_equal'] for x in r['cases'])
new=(ROOT/'candidate_text_math_encoder.py').read_bytes();ast.parse(new.decode())
assert b'VLLM_OMNI_910A_TEXT_FP32_LINEAR' not in new and b'h3_text_audit56' not in new
assert b'get("VLLM_OMNI_910A_TEXT_HF_MATH", "0")' in new
backup=Path(str(target)+'.pre_round56_text_math_20261116.bak')
with backup.open('xb') as f:f.write(old)
target.write_bytes(new);assert target.read_bytes()==new
r={'target':str(target),'backup':str(backup),'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest(),
   'text_math_default_enabled':False,'text_math_optin':'VLLM_OMNI_910A_TEXT_HF_MATH=1','fp32_text_gemm_installed':False,
   'capture_sites_removed':True,'helper_retained_but_unreferenced':prior['helper'],
   'changes':'Model-specific BF16 norm rounding, FP32 outer-product RoPE, embedding-only FP32 promotion and inherited-autocast protection for the existing optional vision linear',
   'scope':'Operator corrections verified; full text/video lossless quality and speed targets are not passed'}
(ROOT/'text_math_deployment.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
