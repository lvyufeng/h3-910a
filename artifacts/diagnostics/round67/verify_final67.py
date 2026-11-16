#!/usr/bin/env python3
"""Read-only final source and selected actual pre-Norm fixture verification."""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io67 import EvidenceWriter

ROOT = Path('/tmp/h3_round67_norm01')
writer = EvidenceWriter(ROOT / 'final_verification.json')
expected = json.loads((ROOT / 'expected_sources.json').read_text())
r = {'production_modified': False, 'production': {}, 'diagnostics': {}, 'captures': []}


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()


for category in ('production', 'diagnostics'):
    for path, digest in expected[category].items():
        observed = sha_file(Path(path))
        r[category][path] = {'sha256': observed, 'expected': digest, 'matches': observed == digest}
r['all_sources_match'] = all(row['matches'] for category in ('production', 'diagnostics') for row in r[category].values())
paths = sorted((ROOT / 'capture').glob('*.pt'))
expected_names = {'%02d.%s.pt' % (layer, kind) for layer in (0, 24, 49) for kind in ('hidden_input', 'hidden_residual', 'q_head', 'k_head')}
assert {path.name for path in paths} == expected_names
for path in paths:
    assert path.is_file() and path.stat().st_size < 100000000
    data = torch.load(path, map_location='cpu', weights_only=True)
    proof = {'path': str(path), 'bytes': path.stat().st_size, 'file_sha256': sha_file(path), 'tensors': {}}
    for name, value in data.items():
        proof['tensors'][name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                                  'sha256': tensor_sha(value), 'finite': bool(torch.isfinite(value.float()).all())}
    proof['output_raw_bits_equal'] = torch.equal(data['native_output'].view(torch.uint8), data['candidate_output'].view(torch.uint8))
    proof['inverse_positive'] = bool(data['inverse_rms'].gt(0).all())
    r['captures'].append(proof)
r['all_selected_actual_outputs_equal_and_finite'] = all(row['output_raw_bits_equal'] and row['inverse_positive']
    and all(value['finite'] for value in row['tensors'].values()) for row in r['captures'])
writer.write(r)
print(json.dumps({'all_sources_match': r['all_sources_match'], 'captures': len(r['captures']),
                  'all_selected_actual_outputs_equal_and_finite': r['all_selected_actual_outputs_equal_and_finite']}, indent=2))
assert r['all_sources_match'] and r['all_selected_actual_outputs_equal_and_finite']
