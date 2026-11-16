#!/usr/bin/env python3
"""Pin executed diagnostic source bytes and restored production identities."""
import hashlib
import json
from pathlib import Path

D=Path('/tmp/h3_round61_adaln01');B=Path('/tmp/h3_round62_blocks01')
OUT=B/'final_source_verification.json'
if OUT.exists():raise RuntimeError('Refuse final source receipt overwrite')
root=Path('/home/mseco/vllm-omni/vllm_omni')
expected={
    'diffusion/models/minimax_h3/minimax_h3_transformer.py':'54e45b10c1650822cb6896df7bb90d107c15d3eea52d47b57a078de64c410d13',
    'diffusion/models/minimax_h3/encoder.py':'7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d',
    'diffusion/attention/backends/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1',
    'diffusion/layers/norm.py':'585f37954fb15feb66c4ed34a8cc9d618fe71b7a9aa4f71747d8ae20ec5dc755',
    'diffusion/distributed/group_coordinator.py':'b674e815a482985f31a0fe8bccf8cb397c8e9575ad7e2d4ec2ac3c97dfc09cb2',
    'platforms/npu/layers/rotary_embedding.py':'5ad286d9d9f18eafb4fa8a864e9d4a43630ae36af98ace5208b42c2069a1cc1d',
}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'production':{},'diagnostic_sources':{},'all_production_unchanged':True,'production_fusion_installed':False}
for name,want in expected.items():
    got=sha(root/name);r['production'][name]={'sha256':got,'expected':want,'unchanged':got==want}
    if got!=want:r['all_production_unchanged']=False
for p in [D/'adaln_vector61.py',D/'point_vector61.py',D/'vec61_launcher.c',D/'libvec61.so',D/'point_guard_validation.py',B/'block_replay_tp8.py']:
    r['diagnostic_sources'][str(p)]=sha(p)
for name in ('diffusion/layers/indexed_modulation.py','diffusion/layers/fused_qk_norm_rope.py','diffusion/attention/layer.py','diffusion/attention/parallel/factory.py'):
    r['production'][name]={'sha256':sha(root/name),'expected':'compare exact retained source snapshot locally'}
assert r['all_production_unchanged'],r['production']
r['scope']='Sources restored after instrumented fixture captures; diagnostic kernels confined to standalone processes and /tmp.'
OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
