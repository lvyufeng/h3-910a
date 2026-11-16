"""Regress retained actual-NPU Norm gates without pretending to emulate TBE."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from summarize import aggregate, summarize

ROOT = Path(__file__).resolve().parent


def rows():
    return [json.loads((ROOT / 'ranks' / ('text_norm_rank%d.json' % rank)).read_text()) for rank in range(8)]


class NormEvidence(unittest.TestCase):
    def test_full_output_identity_and_all_actual_norm_families(self):
        result = summarize()
        self.assertEqual(result['actual_norm_identity_checks'], 1600)
        self.assertEqual(result['actual_residual_identity_checks'], 400)
        self.assertEqual(sum(case['complete_outputs_checked'] for case in result['cases'].values()), 96)
        self.assertFalse(result['whole_service_text_under2s_proven'])
        self.assertFalse(result['official_goal_complete'])
        self.assertFalse(result['production_deployed'])
        for case in result['cases'].values():
            self.assertTrue(case['all_full_outputs_raw_bits_equal'])
            self.assertFalse(case['all_clean_resident_calls_under2s'])
        self.assertGreater(result['speedup_vs_same_run_cached_baseline'], 1)
        self.assertEqual(result['standalone_finite_bf16_word_identity_count'], 65280)

    def test_nonfinite_payload_failure_is_not_hidden_by_finite_subset_pass(self):
        result = summarize()
        proof = json.loads((ROOT / 'validation01.json').read_text())
        self.assertTrue(result['nonfinite_payload_identity_not_accepted'])
        self.assertFalse(proof['all_bf16_words']['all_inputs']['raw_bits_equal'])
        self.assertEqual(result['all_bf16_pattern_native_differences'], 252)
        self.assertTrue(proof['all_bf16_words']['finite_inputs']['raw_bits_equal'])
        self.assertEqual(proof['all_bf16_words']['all_inputs']['elements'], 65536)

    def test_true_both_rounding_boundaries_and_unsafe_gm_shape_guards(self):
        proof = json.loads((ROOT / 'validation02.json').read_text())
        self.assertTrue(proof['completed'])
        self.assertEqual(proof['final_product_boundaries']['exact_midpoints'], 281)
        self.assertEqual(len(proof['cases']), 3)
        self.assertTrue(all(case['raw_bits_equal'] and case['different_raw_words'] == 0 for case in proof['cases']))
        self.assertEqual(len(proof['guards']), 3)
        self.assertTrue(all(guard['rejected'] for guard in proof['guards']))
        self.assertEqual(proof['source_sha256']['norm_tail_guard67.py'], hashlib.sha256((ROOT / 'norm_tail_guard67.py').read_bytes()).hexdigest())
        real_loads = json.loads
        mutations = [lambda row: row['cases'][0].update(name='not_the_executed_fixture'),
                     lambda row: row['cases'][0].update(elements=1),
                     lambda row: row['cases'][0].update(shape=[1, 128]),
                     lambda row: row['guards'][0].update(name='not_an_alignment_guard'),
                     lambda row: row['final_product_boundaries'].update(exact_midpoints=0)]
        for mutation in mutations:
            def changed_loads(text, *args, **kwargs):
                row = real_loads(text, *args, **kwargs)
                if isinstance(row, dict) and 'normalized_boundary_inverse_values' in row:
                    mutation(row)
                return row
            with patch('summarize.json.loads', side_effect=changed_loads):
                with self.assertRaises(ValueError):
                    summarize()

    def test_rank_max_precedes_median_first_call_not_counted(self):
        data = rows()
        for row in data:
            case = row['cases']['native_stats_tail']
            case['first_s_excluded_contains_checks_and_compilation'] = 10000.
            for call in case['warm_calls']:
                call['seconds'] = 1.
        for i in range(3):
            data[i]['cases']['native_stats_tail']['warm_calls'][i]['seconds'] = 10.
        case = aggregate(data)['cases']['native_stats_tail']
        self.assertEqual(case['median_rank_max_s'], 10.)
        self.assertEqual(case['first_rank_max_s_excluded_checks_compilation'], 10000.)

    def test_same_bits_flag_does_not_waive_finite_elements_residual_or_alignment(self):
        mutations = [lambda r: r[0].update(completed=False),
                     lambda r: r[0].update(rank=1),
                     lambda r: r[0].update(parameters_after='mutated'),
                     lambda r: r[0]['cases']['native_stats_tail'].update(tail_enabled=False),
                     lambda r: r[0]['cases']['native_stats_tail']['norm_checks'][0]['output'].update(finite=False),
                     lambda r: r[0]['cases']['native_stats_tail']['norm_checks'][0]['output'].update(elements=1),
                     lambda r: r[0]['cases']['native_stats_tail']['norm_checks'][0].update(aligned_input_gamma_inverse=False),
                     lambda r: r[0]['cases']['native_stats_tail']['norm_checks'].pop(),
                     lambda r: r[0]['cases']['native_stats_tail']['norm_checks'].__setitem__(1, copy.deepcopy(r[0]['cases']['native_stats_tail']['norm_checks'][0])),
                     lambda r: next(check for check in r[0]['cases']['native_stats_tail']['norm_checks'] if 'residual' in check)['residual'].update(raw_bits_equal=False),
                     lambda r: r[0]['cases']['native_stats_tail']['warm_calls'][0]['vs_baseline'].update(max_abs_error=.01),
                     lambda r: r[0]['cases']['native_stats_tail']['shadow_cache'].update(entries=199)]
        for mutation in mutations:
            data = rows()
            mutation(data)
            with self.assertRaises(ValueError):
                aggregate(data)

    def test_selected_actual_tail_fixtures_and_sources_verified_unchanged(self):
        proof = json.loads((ROOT / 'final_verification.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertTrue(proof['all_sources_match'])
        self.assertFalse(proof['production_modified'])
        self.assertEqual(len(proof['production']), 23)
        self.assertEqual(len(proof['diagnostics']), 14)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(proof[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name] for name, value in proof[category].items()))
        self.assertEqual(len(proof['captures']), 12)
        self.assertTrue(proof['all_selected_actual_outputs_equal_and_finite'])
        names = {Path(capture['path']).name for capture in proof['captures']}
        self.assertEqual(names, {'%02d.%s.pt' % (layer, kind) for layer in (0, 24, 49) for kind in ('hidden_input', 'hidden_residual', 'q_head', 'k_head')})
        for capture in proof['captures']:
            self.assertTrue(capture['output_raw_bits_equal'])
            self.assertTrue(capture['inverse_positive'])
            tensors = capture['tensors']
            self.assertEqual(tensors['inverse_rms']['shape'], tensors['x']['shape'][:-1] + [1])
            self.assertEqual(tensors['inverse_rms']['dtype'], 'torch.float32')
            self.assertEqual(tensors['native_output']['sha256'], tensors['candidate_output']['sha256'])
            self.assertTrue(all(value['finite'] for value in tensors.values()))


if __name__ == '__main__':
    unittest.main()
