#!/usr/bin/env python3
"""Read-only hashes of unchanged production plus actual diagnostic dependencies."""
import hashlib
import json
from pathlib import Path
from evidence_io69 import EvidenceWriter

ROOT = Path('/tmp/h3_round69_cast01')
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
