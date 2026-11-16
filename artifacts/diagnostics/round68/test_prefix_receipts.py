import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from summarize import aggregate, summarize

ROOT = Path(__file__).resolve().parent


def rows():
    return [json.loads((ROOT / 'ranks' / ('prefix_rank%d.json' % rank)).read_text()) for rank in range(8)]


class ActualPrefixEvidence(unittest.TestCase):
    def test_all_prefix_statistics_full_outputs_and_scope(self):
        result = summarize()
        self.assertEqual(result['actual_prefix_checks'], 3200)
        self.assertEqual(result['actual_residual_checks'], 800)
        self.assertEqual(sum(case['complete_outputs_checked'] for case in result['cases'].values()), 272)
        self.assertFalse(result['production_deployed'])
        self.assertFalse(result['whole_service_text_under2s_proven'])
        self.assertFalse(result['official_goal_complete'])
        for case in result['cases'].values():
            self.assertFalse(case['all_clean_resident_calls_under2s'])
        self.assertGreater(result['cases']['add_square']['speedup_vs_same_run_native_prefix'], 1)
    def test_square_only_all_word_raw_identity_is_not_numeric_nan_inequality(self):
        result = summarize()
        probe = result['standalone_all_bf16_square_probe']
        self.assertTrue(probe['all_patterns']['raw_bits_equal'])
        self.assertEqual(probe['all_patterns']['elements'], 65536)
        self.assertEqual(probe['all_patterns']['different_values'], 254)
        self.assertFalse(probe['all_patterns']['finite'])
        bounded = probe['bounded_finite_inputs_abs_le_2pow60']
        self.assertTrue(bounded['raw_bits_equal'])
        self.assertTrue(bounded['finite'])
        self.assertEqual(bounded['elements'], 47874)
    def test_readonly_output_format_and_extreme_add_counterexample_retained(self):
        result = summarize()
        self.assertTrue(result['readonly_inputs_and_output_format_guards_passed'])
        self.assertTrue(result['exceptional_add_payload_identity_not_accepted'])
        readonly = json.loads((ROOT / 'validation02.json').read_text())
        self.assertFalse(readonly['exceptional_add_equivalence_claimed'])
        for case in readonly['cases']:
            self.assertTrue(case['x_readonly'])
            self.assertTrue(case['residual_readonly'])
            self.assertTrue(case['outputs_aligned_dense'])
            if case['name'] == 'bounded':
                self.assertTrue(case['square_raw_bits_equal'])
                self.assertTrue(case['updated_raw_bits_equal'])
            if case['name'] == 'extreme_pairs' and case['residual']:
                self.assertFalse(case['square_raw_bits_equal'])
                self.assertFalse(case['updated_raw_bits_equal'])
    def test_wrong_square_variance_inverse_residual_layout_and_final_output_fail(self):
        mutators = [lambda r: r[0].update(rank=1), lambda r: r[0].update(completed=False),
                    lambda r: r[0].update(parameters_after='mutated'),
                    lambda r: r[0]['cases']['square']['prefix_checks'][0]['square'].update(raw_bits_equal=False),
                    lambda r: r[0]['cases']['square']['prefix_checks'][0]['variance'].update(elements=1),
                    lambda r: r[0]['cases']['square']['prefix_checks'][0]['inverse'].update(max_abs_error=1.),
                    lambda r: r[0]['cases']['square']['prefix_checks'][0]['output'].update(finite=False),
                    lambda r: r[0]['cases']['square']['prefix_checks'][0]['square_layout'].__setitem__(3, 29),
                    lambda r: r[0]['cases']['square']['prefix_checks'].pop(),
                    lambda r: next(check for check in r[0]['cases']['add_square']['prefix_checks'] if 'residual' in check)['residual'].update(raw_bits_equal=False),
                    lambda r: r[0]['cases']['add_square']['clean_before'][0]['vs_baseline'].update(elements=1)]
        for change in mutators:
            data = rows()
            change(data)
            with self.assertRaises(ValueError):
                aggregate(data)
    def test_event_vectors_use_same_call_slowest_rank_without_parent_child_sum(self):
        result = summarize()
        compact = json.loads((ROOT / 'breakdown.json').read_text())
        for mode in ('native', 'square', 'add_square'):
            profiles = result['cases'][mode]['profiles']
            selected = sorted(profiles, key=lambda profile: profile['representative']['profile_wall_s'])[1]
            self.assertEqual(compact['modes'][mode]['slowest_wall_rank'], selected['slowest_wall_rank'])
            self.assertEqual(compact['modes'][mode]['profile_call'], selected['profile_call'])
            for profile in profiles:
                slow = profile['slowest_wall_rank']
                self.assertEqual(profile['representative'], profile['all_rank_vectors'][slow])
                for vector in profile['all_rank_vectors']:
                    self.assertEqual(sum(value['count'] for value in vector['categories'].values()), 1103)
                    self.assertIn('child.all_reduce', vector['categories'])
                    self.assertNotIn('top.all_reduce', vector['categories'])
    def test_before_max_rank_then_median_first_and_profile_not_counted(self):
        data = rows()
        for row in data:
            case = row['cases']['add_square']
            case['first_s_excluded_checks_compilation'] = 1000.
            for phase in ('clean_before', 'clean_after', 'profiles'):
                for call in case[phase]:
                    call['seconds'] = 1.
        for i in range(3):
            data[i]['cases']['add_square']['clean_before'][i]['seconds'] = 10.
        self.assertEqual(aggregate(data)['cases']['add_square']['median_clean_before_s'], 10.)
    def test_sources_and_actual_preadd_pairs_verified_without_production_change(self):
        result = json.loads((ROOT / 'final_verification.json').read_text())
        expected = json.loads((ROOT / 'expected_sources.json').read_text())
        self.assertTrue(result['all_sources_match'])
        self.assertFalse(result['production_modified'])
        self.assertEqual(len(result['production']), 23)
        self.assertEqual(len(result['diagnostics']), 15)
        for category in ('production', 'diagnostics'):
            self.assertEqual(set(result[category]), set(expected[category]))
            self.assertTrue(all(value['matches'] and value['sha256'] == expected[category][name] for name, value in result[category].items()))
        self.assertEqual(len(result['captures']), 3)
        self.assertTrue(result['all_actual_preadd_fixtures_valid'])
        for capture in result['captures']:
            self.assertTrue(capture['original_preadd_pair_retained'])
            self.assertTrue(capture['native_inverse_positive'])
            for name in ('original_x', 'original_residual', 'native_updated'):
                self.assertEqual(capture['tensors'][name]['shape'], [1, 2043, 5120])
                self.assertEqual(capture['tensors'][name]['dtype'], 'torch.bfloat16')


if __name__ == '__main__':
    unittest.main()
