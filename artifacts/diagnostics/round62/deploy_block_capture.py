#!/usr/bin/env python3
"""Install/restore capture-only source against verified exact baseline bytes."""
import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--restore',action='store_true');args=p.parse_args()
ROOT=Path('/tmp/h3_round62_blocks01')
TARGET=Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py')
BACKUP=TARGET.with_name(TARGET.name+'.pre_round62_blocks_20261116.bak')
base='54e45b10c1650822cb6896df7bb90d107c15d3eea52d47b57a078de64c410d13'
sha=lambda x:hashlib.sha256(x).hexdigest()
if args.restore:
    old=BACKUP.read_bytes();assert sha(old)==base
    assert TARGET.read_bytes()==(ROOT/'candidate_block_capture.py').read_bytes()
    TARGET.write_bytes(old)
    assert sha(TARGET.read_bytes())==base
    r={'restored':True,'sha256':base,'production_capture_import_removed':True}
    out=ROOT/'capture_restoration.json'
else:
    old=TARGET.read_bytes();assert sha(old)==base
    if BACKUP.exists():raise RuntimeError('Refuse capture backup overwrite')
    BACKUP.write_bytes(old)
    candidate=(ROOT/'candidate_block_capture.py').read_bytes()
    assert 'h3_block_audit62' in candidate.decode()
    TARGET.write_bytes(candidate)
    assert TARGET.read_bytes()==candidate
    r={'installed':True,'baseline_sha256':base,'candidate_sha256':sha(candidate),'backup_path':str(BACKUP),'default_off':True,'forward_math_changed':False}
    out=ROOT/'capture_deployment.json'
if out.exists():raise RuntimeError('Refuse capture receipt overwrite')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
