#!/usr/bin/env python3
"""Read-only hashes of unchanged production plus actual diagnostic dependencies."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, '/tmp/h3_round65_shadow01')
from evidence_io65 import EvidenceWriter

ROOT = Path('/tmp/h3_round66_profile01')
writer = EvidenceWriter(ROOT / 'final_sources.json')
expected = json.loads((ROOT / 'expected_sources.json').read_text())
r = {'production_modified': False, 'production': {}, 'diagnostics': {}}
for category in ('production', 'diagnostics'):
    for path, digest in expected[category].items():
        observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        r[category][path] = {'sha256': observed, 'expected': digest, 'matches': observed == digest}
r['all_sources_match'] = all(row['matches'] for category in ('production', 'diagnostics') for row in r[category].values())
writer.write(r)
print(json.dumps(r, indent=2))
assert r['all_sources_match']
