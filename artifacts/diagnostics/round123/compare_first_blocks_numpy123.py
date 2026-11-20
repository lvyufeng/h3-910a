#!/usr/bin/env python3
"""Same-row GPU/910A first-two-block sample comparison, not full-array error."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from torch_pt_numpy123 import load_tensor_archive
from compare_refiner_numpy123 import metric

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='GPU_first_blocks_reference')
    parser.add_argument('--npu', default='NPU_first_blocks_reference')
    parser.add_argument('--output', default='first_blocks_GPU_vs_NPU123.json')
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    bundle = load_tensor_archive(ROOT / 'same_initial_sampling123.pt')
    result = {'scope': __doc__, 'completed': False, 'metric_scope': '64 complete-channel selected rows per stage per collected SP rank',
              'full_output_finite_range_sha_checks_available_in_receipts': True,
              'full_DiT_step_or_generation_accepted': False, 'whole_tensor_precision_accepted': False,
              'numeric_tolerance': None, 'GPU_reference_denominator': True, 'stages': {}}
    names = ['video_patch_proj', 'audio_patch_proj', 'initial_packed_embeddings_before_SP',
             'block_00_input_local_rows', 'block_00_output_local_rows', 'block_01_input_local_rows', 'block_01_output_local_rows']
    for rank in [0, 2, 4, 6]:
        gpu_root = ROOT / args.gpu
        npu_root = ROOT / args.npu
        gp = gpu_root / f'first_blocks_rank{rank}.json'
        np_ = npu_root / f'first_blocks_rank{rank}.json'
        if not gp.exists() or not np_.exists():
            continue
        gr = json.loads(gp.read_text());nr = json.loads(np_.read_text())
        assert gr['completed'] and nr['completed']
        assert gr['sequence_parallel_rank'] == nr['sequence_parallel_rank']
        assert gr['first_step_unique_timesteps'] == nr['first_step_unique_timesteps']
        assert gr['exact_sampling_payload_sha256'] == nr['exact_sampling_payload_sha256']
        for name in names:
            g = gpu_root / f'{name}_sample_rank{rank}.pt'
            n = npu_root / f'{name}_sample_rank{rank}.pt'
            if not g.exists() or not n.exists():
                continue
            gd = load_tensor_archive(g);nd = load_tensor_archive(n)
            assert gd['full_shape'] == nd['full_shape']
            assert np.array_equal(gd['row_indices'], nd['row_indices'])
            row = {'shape': gd['full_shape'], 'sample_shape': list(gd['value'].shape), 'SP_rank': gr['sequence_parallel_rank'],
                   'dtype': str(gd['value'].dtype), 'sample_all_rows': metric(nd['value'], gd['value']),
                   'GPU_full_tensor_max_abs': gr['captures'][name]['max_abs'],
                   'NPU_full_tensor_max_abs': nr['captures'][name]['max_abs'],
                   'full_tensor_raw_bits_equal': gr['captures'][name]['sha256_raw_tensor_bytes'] == nr['captures'][name]['sha256_raw_tensor_bytes'],
                   'by_sample_token_tag': {}, 'artifact_sha256': {'GPU': hashlib.sha256(g.read_bytes()).hexdigest(), 'NPU': hashlib.sha256(n.read_bytes()).hexdigest()}}
            if 'local_rows' in name or name == 'initial_packed_embeddings_before_SP':
                # SP output hook runs before the capture hook; the historically
                # named "before_SP" capture is actually after SP, proven by shape.
                global_rows = gd['row_indices'] + gr['sequence_parallel_rank'] * gd['full_shape'][0]
                tags = bundle['token_tags'][global_rows]
                for tag in np.unique(tags):
                    mask = tags == tag
                    row['by_sample_token_tag'][str(int(tag))] = metric(nd['value'][mask], gd['value'][mask])
            result['stages'][f'{name}_rank{rank}'] = row
            print(json.dumps({'stage': name, 'rank': rank, 'SP_rank': gr['sequence_parallel_rank'], **row['sample_all_rows']}, indent=2), flush=True)
    result['completed'] = True
    result['all_rank0_stages_compared'] = all(f'{name}_rank0' in result['stages'] for name in names)
    result['capture_label_correction'] = 'initial_packed_embeddings_before_SP is captured after the SP output hook and is [10448,5376] local rows, not the full [41792,5376] tensor.'
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
