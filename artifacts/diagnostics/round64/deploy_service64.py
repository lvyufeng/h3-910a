#!/usr/bin/env python3
"""Hash-pinned temporary service install/rollback, no blind retries."""
import argparse
import hashlib
import json
from pathlib import Path
from h3_evidence64 import EvidenceWriter

ROOT = Path('/tmp/h3_round64_service01')
BASE = Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3')
parser = argparse.ArgumentParser()
parser.add_argument('--restore', action='store_true')
args = parser.parse_args()
receipt = ROOT / ('restoration.json' if args.restore else 'deployment.json')
writer = EvidenceWriter(receipt)
sha = lambda data: hashlib.sha256(data).hexdigest()
items = [('vae.py', 'candidate_vae.py'), ('pipeline_minimax_h3.py', 'candidate_pipeline.py')]
r = {'production': {}, 'new_modules': {}, 'default_off': True, 'global_sdpa_modified': False}
for name, candidate_name in items:
    target = BASE / name
    baseline = (ROOT / 'source' / name).read_bytes()
    candidate = (ROOT / candidate_name).read_bytes()
    backup = BASE / (name + '.pre_round64_20261116.bak')
    assert target.resolve() == target.absolute()
    if args.restore:
        assert target.read_bytes() == candidate and backup.read_bytes() == baseline
        target.write_bytes(baseline)
        assert target.read_bytes() == baseline
    else:
        assert target.read_bytes() == baseline and not backup.exists()
        with backup.open('xb') as f:
            f.write(baseline)
        target.write_bytes(candidate)
        assert target.read_bytes() == candidate
    r['production'][str(target)] = {'baseline_sha256': sha(baseline), 'candidate_sha256': sha(candidate),
                                  'observed_sha256': sha(target.read_bytes()), 'backup': str(backup)}
modules = ('h3_910a_vae.py', 'h3_910a_vae_kernel.py', 'h3_service_audit64.py', 'h3_evidence64.py')
for name in modules:
    target = BASE / name
    content = (ROOT / name).read_bytes()
    assert target.resolve() == target.absolute()
    if args.restore:
        # Retain default-off modules; restored adapter/pipeline no longer import them.
        assert target.read_bytes() == content
    else:
        if target.exists():
            raise FileExistsError('Refuse overwrite of unowned service module: ' + str(target))
        with target.open('xb') as f:
            f.write(content)
    r['new_modules'][str(target)] = sha(content)
r['restored'] = args.restore
r['installed'] = not args.restore
writer.write(r)
print(json.dumps(r, indent=2))
