#!/usr/bin/env python3
"""Read-only source fingerprints after isolated vision factors, CPU only."""
import hashlib
import json
from pathlib import Path
from evidence_io71 import EvidenceWriter

ROOT = Path('/tmp/h3_round71_vision01')
writer = EvidenceWriter(ROOT / 'final_sources.json')
expected = json.loads((ROOT / 'expected_sources.json').read_text())
r = {'production_modified': False, 'production': {}, 'diagnostics': {}}
for category in ('production', 'diagnostics'):
    for path, wanted in expected[category].items():
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        observed = h.hexdigest()
        r[category][path] = {'sha256': observed, 'expected': wanted, 'matches': observed == wanted}
r['all_sources_match'] = all(value['matches'] for category in ('production', 'diagnostics') for value in r[category].values())
writer.write(r)
print(json.dumps({'all_sources_match': r['all_sources_match'], 'production_sources': len(r['production']), 'diagnostic_sources': len(r['diagnostics'])}, indent=2))
assert r['all_sources_match']
