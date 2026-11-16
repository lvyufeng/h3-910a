#!/usr/bin/env python3
"""Verify versioned VAE receipts and summarize only their measured scope."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
EXPECTED_VIDEO_SHAPE = [1, 3, 124, 768, 1344]
EXPECTED_VIDEO_BYTES = 383975424
MODES = ('baseline_c128', 'candidate_c256', 'candidate_c256_linear_fp16', 'baseline_restored')


def load(path):
    return json.loads(path.read_text())


def source_matches(receipt, source):
    return receipt['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()


def aggregate_decode(rows, modes=MODES):
    """Use median(max_rank(call)), never max_rank(median(call)) or cold calls."""
    if len(rows) != 8 or {row['rank'] for row in rows} != set(range(8)):
        raise ValueError('Need eight unique decoder rank receipts')
    if any(not row.get('completed') or row.get('failure') or row.get('cleanup_errors') for row in rows):
        raise ValueError('An incomplete/failed replay cannot be accepted')
    refs = {row['reference_output_sha256'] for row in rows}
    sources = {row['source_sha256'] for row in rows}
    if len(refs) != 1 or len(sources) != 1:
        raise ValueError('All ranks must use identical source and reference')
    reference_sha = refs.pop()
    result = {'rank_count': 8, 'cases': {},
              'measurement': 'median over three calls of per-call rank maximum; barrier start, NPU drain end; first call excluded',
              'scope': 'Complete resident video VAE decode on one actual normalized FL2VA latent. Not service decode stage/audio/MP4 or full 49-step generation quality.'}
    for mode in modes:
        cases = [row['cases'][mode] for row in rows]
        if any(len(case['warm_calls']) != 3 for case in cases):
            raise ValueError('Need three warmed calls on every rank')
        wall = [max(case['warm_calls'][i]['wall_s'] for case in cases) for i in range(3)]
        stream = [max(case['warm_calls'][i]['stream_s'] for case in cases) for i in range(3)]
        calls = [call for case in cases for call in [case['first_call'], *case['warm_calls']]]
        exact = all(call['output']['sha256'] == reference_sha for call in calls)
        expected_contract = all(call['output']['shape'] == EXPECTED_VIDEO_SHAPE
                                and call['output']['dtype'] == 'torch.uint8'
                                and call['output']['bytes'] == EXPECTED_VIDEO_BYTES for call in calls)
        repeats = all(len({call['output']['sha256'] for call in [case['first_call'], *case['warm_calls']]}) == 1
                      for case in cases)
        result['cases'][mode] = {'per_call_rank_max_wall_s': wall, 'median_rank_max_wall_s': statistics.median(wall),
                                 'per_call_rank_max_stream_s': stream, 'median_rank_max_stream_s': statistics.median(stream),
                                 'first_call_rank_max_wall_s': max(case['first_call']['wall_s'] for case in cases),
                                 'all_calls_capture_exact': exact, 'expected_full_video_contract': expected_contract,
                                 'all_repeats_exact': repeats, 'accepted_for_this_captured_latent': exact and expected_contract and repeats,
                                 'warm_calls_all_under_20s_resident_only': all(t < 20 for t in wall)}
    base = result['cases']['baseline_c128']['median_rank_max_wall_s']
    restored = result['cases']['baseline_restored']['median_rank_max_wall_s']
    result['baseline_restored_relative_timing_drift'] = restored / base - 1
    for mode in modes:
        result['cases'][mode]['speedup_vs_first_baseline'] = base / result['cases'][mode]['median_rank_max_wall_s']
    return result


def summarize(root=ROOT):
    tile = load(root / 'tile_replay03.json')
    attention = load(root / 'attention_probe02.json')
    explore = load(root / 'attention_probe03.json')
    for receipt, filename in ((tile, 'tile_replay03.py'), (attention, 'attention_probe02.py'),
                              (explore, 'attention_probe03.py')):
        if not receipt.get('completed') or not source_matches(receipt, root / filename):
            raise ValueError('Incomplete/source-mismatched proof: ' + filename)
    if not tile['capture_baseline_exact'] or not tile['half_constructed_baseline']['vs_actual']['raw_bits_equal']:
        raise ValueError('Cannot call a mismatched standalone decoder the production baseline')
    old = load(root / 'tile_replay02.json')
    result = {
        'scope': 'Round63 actual VAE diagnosis, local integration and clean resident decode; no official lossless/E2E completion.',
        'loader_contract': {'default_fp32_construction_matches_capture': old['capture_baseline_exact'],
                            'default_half_then_fp32_storage_matches_capture': tile['capture_baseline_exact'],
                            'final_parameter_dtypes': tile['final_parameter_dtypes'],
                            'note': 'Final FP32 parameter metadata does not undo the ambient half construction/load-state rounding.'},
        'actual_tile': {'shape': [1, 24, 7, 16, 16], 'decoder_layers': 36, 'output_elements': 5505024,
                        'cases': {key: {'median_warm_s': value['median_s'],
                                        'all_repeats_equal': value['all_repeats_equal'],
                                        'raw_bits_equal_to_actual_capture': value['vs_actual']['raw_bits_equal']}
                                  for key, value in tile['cases'].items()}},
        'actual_attention': {'shape_bshd': [1, 1797, 32, 64],
                             'baseline_capture_exact': attention['cases']['shim_original_c128']['vs_actual']['raw_bits_equal'],
                             'baseline_ms': attention['cases']['shim_original_c128']['timing']['median_ms'],
                             'chunk256_ms': attention['cases']['shim_original_c256']['timing']['median_ms'],
                             'custom_fa_ms_separate_card_run': explore['cases']['custom_fa_exploratory']['timing']['median_ms'],
                             'custom_fa_vs_capture': explore['cases']['custom_fa_exploratory']['vs_actual'],
                             'custom_fa_vs_fp32_oracle': explore['cases']['custom_fa_exploratory']['vs_fp32_oracle'],
                             'legacy_vs_fp32_oracle': explore['native_capture_vs_fp32_oracle'],
                             'custom_fa_lossless_accepted': False},
        'production_fast_path_installed': False,
        'official_quality_or_task_matched_comparison_achieved': False,
        'strict_goal_complete': False,
    }
    paths = sorted((root / 'decode_replay').glob('decode_replay01_rank*.json'))
    if paths:
        rows = [load(path) for path in paths]
        if any(not source_matches(row, root / 'decode_replay_tp8.py') for row in rows):
            raise ValueError('Decoder replay source bytes differ from executed receipts')
        result['complete_resident_decoder'] = aggregate_decode(rows)
    else:
        result['complete_resident_decoder'] = {'not_yet_collected': True}
    tiling = load(root / 'attention_fa_tiling01.json')
    if not source_matches(tiling, root / 'attention_fa_tiling.py'):
        raise ValueError('FA tiling source mismatch')
    result['exploratory_raw_fa_tiling'] = {'cases': tiling['cases'], 'failure_retained': tiling.get('failure'),
                                           'raw_kernel_only_not_wrapper_ms': True, 'lossless_accepted': False}
    packed = load(root / 'pack_validation01.json')
    for filename, digest in packed['sources'].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Packing proof source mismatch: ' + filename)
    result['packing_experiment'] = {'pack_bits_equal': packed['pack_vs_torch']['raw_bits_equal'],
                                    'unpack_bits_equal': packed['unpack_vs_k']['raw_bits_equal'],
                                    'original_fa_wrapper_ms': packed['cases']['existing_wrapper_br128_bc256']['timing']['median_ms'],
                                    'fused_pack_same_fa_wrapper_ms': packed['cases']['packed_wrapper_br128_bc256']['timing']['median_ms'],
                                    'same_kernel_fa_output_bits_equal': packed['cases']['packed_wrapper_br128_bc256']['vs_existing_same_tile_fa']['raw_bits_equal'],
                                    'material_latency_benefit_demonstrated': False,
                                    'scope': 'Prepared device must stay current; finite captured outputs only; not arbitrary nonfinite FA behavior.'}
    sanitizer = load(root / 'sanitize_validation02.json')
    if not sanitizer.get('completed'):
        raise ValueError('Sanitizer operator proof incomplete')
    for filename, digest in sanitizer['sources'].items():
        if hashlib.sha256((root / filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Sanitizer proof source mismatch: ' + filename)
    result['tbe_finite_select'] = {'all_65536_half_words_match_integer_oracle': sanitizer['exhaustive']['vs_integer_bit_oracle']['raw_bits_equal'],
                                  'finite_words_match_configured_native': sanitizer['exhaustive']['finite_words_vs_native']['raw_bits_equal'],
                                  'configured_native_is_identity_on_all_half_words': sanitizer['exhaustive']['native_output_vs_unfiltered_words']['raw_bits_equal'],
                                  'nonfinite_words_with_deliberately_different_handling': 2048,
                                  'actual_capture_raw_nonfinite_count': sanitizer['actual_raw_output_nonfinite_count_cpu'],
                                  'cases': sanitizer['cases'],
                                  'scope': 'Documented finite-select semantics; not native equivalence on NaN/Inf. Actual finite fixture is bit-exact.'}
    paths = sorted((root / 'decode_sanitize').glob('decode_sanitize01_rank*.json'))
    if paths:
        rows = [load(path) for path in paths]
        if any(not source_matches(row, root / 'decode_sanitize_tp8.py') for row in rows):
            raise ValueError('Full sanitizer replay source mismatch')
        result['complete_resident_decoder_with_tbe_finite_select'] = aggregate_decode(
            rows, modes=('baseline_c128', 'sanitize_c128', 'sanitize_c256', 'baseline_restored'))
    else:
        result['complete_resident_decoder_with_tbe_finite_select'] = {'not_yet_collected': True}
    return result


if __name__ == '__main__':
    result = summarize()
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
