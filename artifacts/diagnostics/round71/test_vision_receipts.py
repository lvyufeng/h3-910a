"""Acceptance regressions against actual vision factor receipts, never synthetic numerics."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from summarize_vision71 import aggregate, expected_modules, MAX_BYTES

ROOT = Path(__file__).resolve().parent


class ActualVisionFactors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = [ROOT / 'ranks' / ('vision_rank%d.json' % rank) for rank in range(8)]
        if not all(path.is_file() for path in paths):
            raise unittest.SkipTest('Actual eight-rank NPU receipt collection is not complete')
        cls.rows = [json.loads(path.read_text()) for path in paths]
        cls.previous = [json.loads((ROOT.parent / 'round70' / 'ranks' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    def test_actual_full_encoder_bit_identity_and_experimental_scope(self):
        report = aggregate(self.rows, self.previous)
        self.assertTrue(report['complete_output_identity_accepted'])
        self.assertEqual(report['complete_outputs_compared'], 160)
        self.assertEqual(report['actual_vision_linear_checks'], 2784)
        self.assertEqual(report['actual_input_cast_checks'], 2784)
        self.assertEqual(report['actual_weight_and_bias_checks'], 5568)
        self.assertEqual(report['enabled_activation_cast_checks'], 1856)
        self.assertEqual(report['native_activation_control_comparisons'], 928)
        self.assertEqual(report['enabled_weight_and_bias_shadow_checks'], 3712)
        self.assertEqual(report['native_weight_and_bias_control_comparisons'], 1856)
        self.assertFalse(report['whole_service_text_under2s_proven'])
        self.assertFalse(report['official_goal_complete'])
        self.assertFalse(report['production_deployed'])
    def test_geometry_and_independent_full_shadow_factor(self):
        report = aggregate(self.rows, self.previous)
        self.assertEqual(len(expected_modules()), 116)
        for name, case in report['cases'].items():
            self.assertEqual(case['vision_shadow_bytes_per_rank'], MAX_BYTES if name in ('both', 'shadow_only') else 0)
        self.assertEqual(report['cases']['restored']['vision_shadow_bytes_per_rank'], 0)
    def test_exact_input_weight_bias_output_and_layout_proofs_cannot_be_waived(self):
        for mutation in ('bits', 'weight_elements', 'bias_error', 'input_format', 'output_stride', 'readonly', 'missing_layer'):
            rows = deepcopy(self.rows)
            case = rows[0]['cases']['both']; check = case['vision_linear_checks'][0]
            if mutation == 'bits': check['input_cast']['raw_bits_equal'] = False
            elif mutation == 'weight_elements': check['weight_shadow']['elements'] -= 1
            elif mutation == 'bias_error': check['bias_shadow']['max_abs_error'] = 0.5
            elif mutation == 'input_format': check['input_cast_layout']['physical_format'] += 1
            elif mutation == 'output_stride': check['output_layout']['stride'][0] += 1
            elif mutation == 'readonly': check['input_readonly'] = False
            else: case['vision_linear_checks'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows, self.previous)
    def test_wrong_cache_occupancy_hits_or_bypass_and_unreleased_wrapper_fail(self):
        for mutation in ('cache_bytes', 'entries', 'hits', 'bypasses', 'restore'):
            rows = deepcopy(self.rows)
            cache = rows[0]['cases']['both']['vision_shadow_cache']
            if mutation == 'cache_bytes': cache['cached_fp32_bytes'] -= 1
            elif mutation == 'entries': cache['entries'] = 0
            elif mutation == 'hits': cache['hits'] = 0
            elif mutation == 'bypasses': cache['uncacheable_bypasses'] = 1
            else: rows[0]['vision_wrappers_restored'] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows, self.previous)
    def test_changed_runtime_flags_or_fixed_text_mode_rejected(self):
        for mutation in ('math_flag', 'text_mode', 'vision_flag'):
            rows = deepcopy(self.rows)
            if mutation == 'math_flag': rows[0]['math_flags_after']['matmul_allow_hf32'] = not rows[0]['math_flags_before']['matmul_allow_hf32']
            elif mutation == 'text_mode': rows[0]['cases']['both']['fixed_text_mode'] = 'native_text'
            else: rows[0]['environment']['VLLM_OMNI_910A_VISION_FP32'] = '0'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows, self.previous)
    def test_changed_fresh_vision_is_preserved_counterexample_not_full_stage_acceptance(self):
        rows = deepcopy(self.rows)
        rows[0]['cases']['both']['warm_calls'][0]['identities']['vision_output'][0]['sha256'] = '0' * 64
        report = aggregate(rows, self.previous)
        self.assertFalse(report['intermediate_identity_against_round70']['vision_output'])
        self.assertFalse(report['complete_output_identity_accepted'])
        self.assertFalse(report['complete_encode_ids_under2s_accepted'])
    def test_three_warm_call_rank_max_precedes_median_and_excluded_first(self):
        report = aggregate(self.rows, self.previous)
        for name in report['cases']:
            values = sorted(max(row['cases'][name]['warm_calls'][i]['seconds'] for row in self.rows) for i in range(3))
            self.assertEqual(report['cases'][name]['median_complete_encode_ids_s'], values[1])
        self.assertEqual(report, aggregate(list(reversed(self.rows)), self.previous))


if __name__ == '__main__':
    unittest.main()
