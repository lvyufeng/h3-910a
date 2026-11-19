#!/usr/bin/env python3
"""Regenerate manifest107.json: file hashes, geometry, padding and memory contract.

Standard library only; imports the adapter (which never imports torch at module scope).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
import importlib.util

_spec = importlib.util.spec_from_file_location('vision_tp_adapter107', HERE / 'vision_tp_adapter107.py')
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    geometry = mod.padded_geometries()
    per_block_elems = sum(item['n'] * item['k'] for item in geometry.values())
    test_source = (HERE / 'test_vision_tp_adapter107.py').read_text()
    declared_tests = sum(1 for line in test_source.splitlines()
                         if line.strip().startswith('def test_'))
    adapter_lines = len((HERE / 'vision_tp_adapter107.py').read_text().splitlines())
    manifest = {
        'round': 107,
        'artifact': 'tp_vision',
        'scope': ('SOURCE ONLY: private original-Half Vision TP8 adapter for the 27 Qwen3-VL '
                  'vision blocks (attention + MLP). Not compiled, not launched, not measured '
                  'by its author; the root agent owns every hardware receipt.'),
        'source_only': True,
        'compiled_by_author': False,
        'executed_by_author': False,
        'hardware_receipts_by_author': 0,
        'files': {
            'vision_tp_adapter107.py': sha(HERE / 'vision_tp_adapter107.py'),
            'test_vision_tp_adapter107.py': sha(HERE / 'test_vision_tp_adapter107.py'),
            'make_manifest107.py': sha(HERE / 'make_manifest107.py'),
        },
        'api': {
            'constructor': 'VisionTPAdapter107(candidate, group, backend, attention, m=8064)',
            'methods': ['prime(vision_or_encoder)', 'set_enabled(bool)', 'snapshot()',
                        'uninstall()'],
            'backend_contract': {
                'prime': 'prime(name, weight_half, bias_half_or_None, mode, m=8064) -> entry',
                'run': ('run(entry, activation_half_2d) -> Half (m,n) in column mode or '
                        'Float32 (m,n) row partial in row mode'),
                'snapshot': 'snapshot() -> data (recorded as an opaque backend receipt)',
            },
            'attention_contract': ('standalone callable (q, k, v) -> Half 4D, called once per '
                                   'image segment with strict shape (1, 2, 4032, 72); it owns '
                                   'the original 72 ** -0.5 scale and is never installed as a '
                                   'global helper'),
            'group_contract': ('rank_in_group, world_size == 8 and '
                               'all_reduce(Float32Tensor) -> Float32Tensor'),
        },
        'patch_surface': {
            'patched': ['blocks[i].attn.forward', 'blocks[i].mlp.forward'],
            'mechanism': 'instance attribute only (types.MethodType), never a class-level patch',
            'count': 54,
            'untouched': ['patch_embed', '4 merger families', 'LayerNorm',
                          'rotary geometry', 'cu_seqlens construction', 'DeepStack',
                          '50 text layers', '_NpuBF16CompatibleLinear.forward'],
            'conflict_note': ('VisionBiasAdapter87 patches the Linear CLASS; this adapter bypasses '
                              'the 108 block linears while enabled and leaves the 8 merger '
                              'linears to that existing adapter'),
        },
        'geometry': {
            'm': mod.VISION_TOKENS,
            'segments': mod.VISION_TOKENS // mod.VISION_SEGMENT,
            'segment_tokens': mod.VISION_SEGMENT,
            'hidden': mod.VISION_DIM,
            'heads': mod.VISION_HEADS,
            'head_dim': mod.VISION_HEAD_DIM,
            'mlp_intermediate': mod.VISION_MLP,
            'blocks': mod.VISION_BLOCKS,
            'tp_world': mod.TP_WORLD,
            'local_heads': mod.VISION_HEADS // mod.TP_WORLD,
            'local_head_channels': mod.VISION_DIM // mod.TP_WORLD,
            'local_mlp_channels': mod.VISION_MLP // mod.TP_WORLD,
            'channel_coverage': '4304 = 8*538 and 1152 = 8*144 tile exactly; no gap or overlap',
            'm_is_cube_aligned': mod.VISION_TOKENS % mod.CUBE_ALIGN == 0,
        },
        'slices': {str(rank): {
            'qkv_row_slices': mod.plan_local_geometry(rank)['qkv_row_slices'],
            'out_proj_input_columns': mod.plan_local_geometry(rank)['out_proj_input_columns'],
            'mlp_output_rows_and_fc2_input_columns': mod.plan_local_geometry(rank)['mlp_channel_span'],
        } for rank in range(mod.TP_WORLD)},
        'backend_geometries': geometry,
        'dtype_contract': {
            'original_weights_and_biases': 'Half, untouched, on npu:<rank>',
            'column_mode_output': 'Half after Float32 accumulation + exact Half bias widened to Float32',
            'row_mode_output': 'Float32 partial with NO bias; collective in Float32; then full '
                               'original bias widened to Float32; then exactly one Half cast',
            'module_boundary_output': 'Half (m, 1152), so the next LayerNorm sees the same dtype',
            'activation_into_gelu': 'Half (m, 538): the original nn.GELU(approximate="tanh") runs '
                                    'only on real logical channels, never on Cube padding',
            'forbidden': ['Float weight shadow', 'BF16 gold cast', 'cached activations',
                          'cached hidden conditioning', 'fake final norm', 'clamp',
                          'numerical tolerance', 'default promotion', '8x speedup claim'],
        },
        'expected_counters_per_vision_forward': mod.expected_counts_per_forward(),
        'memory': {
            'adapter_cached_tensors': 0,
            'adapter_cached_activations': 0,
            'half_weight_slice_elements_per_block': per_block_elems,
            'half_weight_slice_bytes_per_block_transient': per_block_elems * 2,
            'half_weight_slice_bytes_all_27_blocks_if_backend_stores_them': per_block_elems * 2 * mod.VISION_BLOCKS,
            'float_bias_widening_bytes_per_collective_output': mod.VISION_DIM * 4,
            'note': ('the adapter hands each Half slice to backend.prime() and drops its own '
                     'reference; any retained packing is the backend\'s, reported there'),
        },
        'root_must_verify_on_hardware': [
            'Cube support for m=8064 with the four (k,n) shapes and the 128-padded k/n',
            'column mode really accumulates in Float32 over the original Half input/weight and '
            'applies the exact Half bias widened to Float32 before one Half cast',
            'row mode really returns a Float32 partial (a Half partial must fail closed here)',
            'all_reduce stays Float32 and returns the same shape',
            'the 2-head local attention callable returns Half (1,2,4032,72) per image segment',
            'the backend pads k/n to k_pad/n_pad and returns only the logical n columns',
            'cu_seqlens is [0,4032,8064] for the two benchmark images',
            'enabled forward reproduces 27 attention + 27 MLP calls, 54 collectives, 108 primed '
            'linear weights and 54 local attention invocations',
            'disable restores the native full-16-head baseline bit-for-bit',
        ],
        'no_actual_claim': {
            'numerical_tolerance': None,
            'full_encoder_precision_accepted': False,
            'whole_goal_complete': False,
            'eight_times_speedup_claimed': False,
            'author_hardware_runs': 0,
            'why': ('Float32 HCCL partials can change low bits and fewer local heads change the '
                    'attention grouping; only the root real graph against the original CPU FP16 '
                    'reference can accept or reject this path'),
        },
        'lines': {'vision_tp_adapter107.py': adapter_lines},
        'tests': {
            'file': 'test_vision_tp_adapter107.py',
            'count': declared_tests,
            'framework': 'python -m unittest (standard library)',
            'imports_real_torch': False,
            'uses_record_only_fake_tensors': True,
        },
    }
    out = HERE / 'manifest107.json'
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'written': str(out), 'files': manifest['files'],
                      'tests': manifest['tests']['count'],
                      'half_weight_slice_bytes_all_27_blocks_if_backend_stores_them':
                          manifest['memory']['half_weight_slice_bytes_all_27_blocks_if_backend_stores_them']},
                     indent=2))


if __name__ == '__main__':
    main()