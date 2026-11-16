#!/usr/bin/env python3
"""Read-only final source verification after stopping/restoring our service."""
import hashlib
import json
from pathlib import Path
from h3_evidence64 import EvidenceWriter

ROOT = Path('/tmp/h3_round64_service01')
BASE = Path('/home/mseco/vllm-omni/vllm_omni/diffusion/models/minimax_h3')
writer = EvidenceWriter(ROOT / 'final_verification.json')
expected = json.loads((ROOT / 'expected_final.json').read_text())
r = {'production': {}, 'diagnostics': {}, 'default_off': True, 'service_engine_intentionally_left_running': False}
for category in ('production', 'diagnostics'):
    for path, digest in expected[category].items():
        observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        r[category][path] = {'sha256': observed, 'expected': digest, 'matches': observed == digest}
r['all_sources_match'] = all(row['matches'] for values in (r['production'], r['diagnostics']) for row in values.values())
r['original_adapter_and_pipeline_restored'] = all(r['production'][str(BASE / filename)]['matches'] for filename in ('vae.py', 'pipeline_minimax_h3.py'))
writer.write(r)
print(json.dumps(r, indent=2))
assert r['all_sources_match'] and r['original_adapter_and_pipeline_restored']
