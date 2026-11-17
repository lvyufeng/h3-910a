#!/usr/bin/env python3
"""Temporary hash-pinned VAE-only opt-in install/restore for actual generation."""
import argparse
import hashlib
from pathlib import Path
from evidence_io76 import EvidenceWriter

ROOT=Path('/tmp/h3_round76_deploy01')
TARGET=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py')
BACKUP=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/vae.py.pre_round76_20261117.bak')
p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');args=p.parse_args()
writer=EvidenceWriter(ROOT/('restoration.json'if args.restore else'deployment.json'))
sha=lambda content:hashlib.sha256(content).hexdigest()
before=(ROOT/'vae_original.py').read_bytes();candidate=(ROOT/'candidate_vae76.py').read_bytes()
assert sha(before)=='4d5bbe4ff0f4b5d00757682b76465a16356aff5cc28b40f3caeb53e858731827'
assert TARGET.resolve()==TARGET.absolute()and BACKUP.resolve()==BACKUP.absolute()
if args.restore:
    assert TARGET.read_bytes()==candidate and BACKUP.read_bytes()==before
    TARGET.write_bytes(before);assert TARGET.read_bytes()==before
else:
    assert TARGET.read_bytes()==before and not BACKUP.exists()
    with BACKUP.open('xb')as f:f.write(before)
    TARGET.write_bytes(candidate);assert TARGET.read_bytes()==candidate
r={'baseline_sha256':sha(before),'candidate_sha256':sha(candidate),'observed_sha256':sha(TARGET.read_bytes()),
   'target':str(TARGET),'backup':str(BACKUP),'restored':args.restore,'default_off':True,'only_VAE_decoder_instance_hook':True,
   'encoder_and_pipeline_sources_modified':False,'global_Torch_or_SDPA_changed_by_this_deployment':False}
writer.write(r)
