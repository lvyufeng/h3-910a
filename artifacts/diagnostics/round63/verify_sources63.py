#!/usr/bin/env python3
"""Read-only source verification; publish only an isolated diagnostic receipt."""
import hashlib
import json
from pathlib import Path
from evidence_io63 import EvidenceWriter

ROOT = Path('/tmp/h3_round63_decode01')
OUT = ROOT / 'final_source_verification.json'
writer = EvidenceWriter(OUT)
expected = json.loads((ROOT / 'expected_sources.json').read_text())
receipt = {'production': {}, 'diagnostic_sources': {}, 'production_fast_path_installed': False}
for category in ('production', 'diagnostic_sources'):
    for name, wanted in expected[category].items():
        observed = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        receipt[category][name] = {'expected': wanted, 'sha256': observed, 'unchanged': observed == wanted}
receipt['all_production_unchanged'] = all(row['unchanged'] for row in receipt['production'].values())
receipt['all_diagnostic_sources_match'] = all(row['unchanged'] for row in receipt['diagnostic_sources'].values())
writer.write(receipt)
print(json.dumps(receipt, indent=2))
assert receipt['all_production_unchanged'] and receipt['all_diagnostic_sources_match']
