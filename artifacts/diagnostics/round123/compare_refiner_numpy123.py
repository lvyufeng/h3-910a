#!/usr/bin/env python3
"""Validate full same-input GPU/910A refiner tensors with NumPy, no model inference."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from torch_pt_numpy123 import load_tensor_archive

ROOT = Path(__file__).resolve().parent


def metric(candidate, reference):
    assert candidate.shape == reference.shape and candidate.dtype == reference.dtype
    a = candidate.astype(np.float64)
    b = reference.astype(np.float64)
    error = a - b
    return {'rows': int(a.shape[0]), 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'relative_RMS': float(np.sqrt(np.sum(error * error) / max(np.sum(b * b), 1e-300))),
            'RMSE': float(np.sqrt(np.mean(error * error))), 'max_abs_error': float(np.max(np.abs(error))),
            'GPU_max_abs': float(np.max(np.abs(b))), 'NPU_max_abs': float(np.max(np.abs(a))),
            'raw_bits_identical': candidate.tobytes() == reference.tobytes(),
            'changed_value_fraction': float(np.mean(candidate != reference))}


def main():
    output = ROOT / 'full_refiner_GPU_vs_NPU123.json'
    assert not output.exists()
    bundle = load_tensor_archive(ROOT / 'same_initial_sampling123.pt')
    paths = {'GPU': ROOT / 'refiner_compare_minimal/GPU_refiner_final_rank0_legacy.pt',
             'NPU': ROOT / 'refiner_compare_minimal/NPU_refiner_final_rank0_legacy.pt'}
    receipts = {'GPU': json.loads((ROOT / 'GPU_refiner_reference/refiner_replay_rank0.json').read_text()),
                'NPU': json.loads((ROOT / 'NPU_refiner_partial/refiner_replay_rank0.json').read_text())}
    values = {name: load_tensor_archive(path)['value'] for name, path in paths.items()}
    for name, tensor in values.items():
        assert tensor.shape == (2043, 5376) and tensor.dtype == np.float16
        assert hashlib.sha256(tensor.tobytes()).hexdigest() == receipts[name]['captures']['refiner_final']['sha256_raw_tensor_bytes']
        assert receipts[name]['completed']
    assert receipts['GPU']['actual_input_sha256'] == receipts['NPU']['actual_input_sha256']
    assert receipts['GPU']['sampling_payload_sha256'] == receipts['NPU']['sampling_payload_sha256']
    tags = bundle['text_tags']
    result = {'scope': __doc__, 'completed': True, 'model_inference_executed_locally': False,
              'same_input_tensor_digest_verified': True, 'full_output_raw_digests_verified': True,
              'teacher_forced_GPU_encoder_input': True, 'rank': 0,
              'shape': [2043, 5376], 'dtype': 'float16', 'GPU_reference_denominator': True,
              'whole_encoder_or_full_DiT_precision_accepted': False, 'numeric_tolerance': None,
              'sampling_payload_sha256': receipts['GPU']['sampling_payload_sha256'],
              'all_rows': metric(values['NPU'], values['GPU']), 'by_token_tag': {}}
    for tag in np.unique(tags):
        selected = tags == tag
        result['by_token_tag'][str(int(tag))] = metric(values['NPU'][selected], values['GPU'][selected])
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
