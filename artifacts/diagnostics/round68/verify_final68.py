#!/usr/bin/env python3
"""Read-only source/actual pre-add fixture verification; run CPU-only."""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io68 import EvidenceWriter

ROOT = Path('/tmp/h3_round68_prefix01')
torch.set_num_threads(4)
writer = EvidenceWriter(ROOT / 'final_verification.json')
expected = json.loads((ROOT / 'expected_sources.json').read_text())
r = {'production_modified': False, 'production': {}, 'diagnostics': {}, 'captures': []}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            data = f.read(8 * 1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


for category in ('production', 'diagnostics'):
    for path, digest in expected[category].items():
        value = sha_file(Path(path))
        r[category][path] = {'sha256': value, 'expected': digest, 'matches': value == digest}
r['all_sources_match'] = all(value['matches'] for category in ('production', 'diagnostics') for value in r[category].values())
paths = sorted((ROOT / 'capture').glob('*.pt'))
assert {path.name for path in paths} == {'%02d.hidden_residual.pt' % layer for layer in (0, 24, 49)}
for path in paths:
    assert path.is_file() and path.stat().st_size < 100000000
    values = torch.load(path, map_location='cpu', weights_only=True)
    assert set(values) == {'original_x', 'original_residual', 'native_updated', 'native_variance', 'native_inverse'}
    row = {'path': str(path), 'file_sha256': sha_file(path), 'file_bytes': path.stat().st_size, 'tensors': {}}
    for name, value in values.items():
        dense = value.contiguous()
        row['tensors'][name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                                'sha256': hashlib.sha256(memoryview(dense.view(torch.uint8).numpy())).hexdigest(),
                                'finite': bool(torch.isfinite(value.float()).all()), 'max_abs': float(value.float().abs().max())}
    row['original_preadd_pair_retained'] = values['original_x'].shape == values['original_residual'].shape == values['native_updated'].shape == (1, 2043, 5120)
    row['native_inverse_positive'] = bool(values['native_inverse'].gt(0).all())
    r['captures'].append(row)
r['all_actual_preadd_fixtures_valid'] = all(row['original_preadd_pair_retained'] and row['native_inverse_positive']
    and all(value['finite'] for value in row['tensors'].values()) for row in r['captures'])
writer.write(r)
print(json.dumps({'all_sources_match': r['all_sources_match'], 'captures': len(r['captures']),
                  'all_actual_preadd_fixtures_valid': r['all_actual_preadd_fixtures_valid']}, indent=2))
assert r['all_sources_match'] and r['all_actual_preadd_fixtures_valid']
