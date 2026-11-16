#!/usr/bin/env python3
"""Read-only final source and fresh multimodal-reference verification; CPU-only."""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io70 import EvidenceWriter

ROOT = Path('/tmp/h3_round70_encoder01')
COMPONENT_ROOT = Path('/tmp/h3_round70_components01')
writer = EvidenceWriter(ROOT / 'final_verification.json')
expected = json.loads((ROOT / 'expected_sources.json').read_text())
r = {'production_modified': False, 'production': {}, 'diagnostics': {}, 'fresh_reference': {}}
torch.set_num_threads(4)


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.numel() * value.element_size(),
                'sha256': hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest(),
                'finite': bool(torch.isfinite(value.float()).all())}
    if isinstance(value, (list, tuple)):
        return [identity(item) for item in value]
    if isinstance(value, dict):
        return {name: identity(item) for name, item in value.items()}
    return value


for category in ('production', 'diagnostics'):
    for path, digest in expected[category].items():
        observed = sha_file(Path(path))
        r[category][path] = {'sha256': observed, 'expected': digest, 'matches': observed == digest}
r['all_sources_match'] = all(value['matches'] for category in ('production', 'diagnostics') for value in r[category].values())
path = COMPONENT_ROOT / 'fresh_encoder_reference.pt'
assert path.is_file() and path.stat().st_size < 512 * 1024 * 1024
values = torch.load(path, map_location='cpu', weights_only=True)
assert set(values) == {'embedding_output', 'vision_output', 'text_inputs', 'complete_output'}
proof = json.loads((COMPONENT_ROOT / 'encoder_rank0.json').read_text())['cases']['baseline']['first_identities']
measured = identity(values)
r['fresh_reference'] = {'path': str(path), 'bytes': path.stat().st_size, 'file_sha256': sha_file(path),
                         'tensors': measured, 'matches_actual_first_baseline': all(measured[name] == proof[name] for name in measured),
                         'scope': 'Fresh real BF16 encoder, original FP16 transport pixels, VISION_FP32=1, experimental text math. This is a replay fixture, NOT an independent HF oracle.'}
writer.write(r)
print(json.dumps({'all_sources_match': r['all_sources_match'], 'fresh_reference_matches_actual_first_baseline': r['fresh_reference']['matches_actual_first_baseline'],
                  'fresh_reference_bytes': r['fresh_reference']['bytes']}, indent=2))
assert r['all_sources_match'] and r['fresh_reference']['matches_actual_first_baseline']
