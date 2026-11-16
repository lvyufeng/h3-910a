"""Acceptance regressions using real complete encoder and component receipts."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from summarize import aggregate, tensor_contract
from summarize_components import aggregate_components

ROOT = Path(__file__).resolve().parent


class AcceptanceRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = [json.loads((ROOT / 'ranks' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
        cls.probes = [json.loads((ROOT / 'components' / ('encoder_rank%d.json' % rank)).read_text()) for rank in range(8)]
    def test_actual_complete_inputs_match_without_accepting_service_or_two_seconds(self):
        report = aggregate(self.main)
        self.assertTrue(report['complete_output_identity_accepted'])
        self.assertEqual(report['complete_outputs_compared'], 96)
        self.assertEqual(report['actual_norm_checks'], 1600)
        self.assertEqual(report['actual_residual_checks'], 400)
        self.assertEqual(report['actual_cast_checks'], 1600)
        self.assertEqual(report['actual_projection_checks'], 2400)
        self.assertAlmostEqual(report['cases']['fast']['median_complete_encode_ids_s'], 7.0500054997392)
        for key in ('complete_encode_ids_under2s_accepted', 'whole_service_text_under2s_proven', 'official_goal_complete', 'production_deployed'):
            self.assertFalse(report[key])
    def test_actual_probe_requires_fresh_main_inputs_and_preserves_under2s_false(self):
        report = aggregate_components(self.probes, self.main)
        self.assertTrue(report['complete_output_identity_accepted'])
        self.assertEqual(report['complete_outputs_compared'], 240)
        self.assertEqual(report['full_encode_ids_output_sha256'], '67ef196e33bcaa495420da1ebc07d1ff6dcaf89ddbaf75a350940ed8ae210eee')
        self.assertFalse(report['complete_encode_ids_under2s_accepted'])
        self.assertFalse(report['whole_service_text_under2s_proven'])
    def test_nonhex_hash_missing_vision_and_wrong_cpu_output_shape_rejected(self):
        value = deepcopy(self.main[0]['actual_inputs_before']['input_ids'])
        value['sha256'] = 'z' * 64
        with self.assertRaises(ValueError):
            tensor_contract(value, [2043], 'torch.int64', 16344)
        for mutation in ('missing_vision', 'cpu_shape'):
            rows = deepcopy(self.main)
            ids = rows[0]['cases']['fast']['warm_calls'][0]['identities']
            if mutation == 'missing_vision':
                ids.pop('vision_output')
            else:
                ids['complete_output']['shape'] = [1, 2043, 5120]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows)
    def test_precision_flags_and_true_parameter_footprint_are_pinned(self):
        for mutation in ('vision_fp32', 'hfmath', 'embedding', 'fp16source'):
            rows = deepcopy(self.main)
            if mutation == 'vision_fp32': rows[0]['environment']['VLLM_OMNI_910A_VISION_FP32'] = '0'
            elif mutation == 'hfmath': rows[0]['environment']['VLLM_OMNI_910A_TEXT_HF_MATH'] = '0'
            elif mutation == 'embedding': rows[0]['component_parameter_bytes']['embedding'] = 0
            else: rows[0]['checkpoint_root'] = '/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/text_encoder'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows)
    def test_nonzero_norm_error_or_missing_cast_proofs_cannot_pass(self):
        for mutation in ('norm', 'cast'):
            rows = deepcopy(self.main)
            case = rows[0]['cases']['fast']
            if mutation == 'norm': case['norm_checks'][0]['output']['max_abs_error'] = 1.
            else: case['cast_checks'].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate(rows)
    def test_per_call_changed_fresh_vision_is_counterexample_not_accepted_encoder(self):
        rows = deepcopy(self.main)
        rows[0]['cases']['fast']['warm_calls'][0]['identities']['vision_output'][0]['sha256'] = '0' * 64
        report = aggregate(rows)
        self.assertFalse(report['intermediate_identity']['vision_output'])
        self.assertFalse(report['complete_output_identity_accepted'])
        self.assertFalse(report['complete_encode_ids_under2s_accepted'])
    def test_final_numerical_difference_is_preserved_as_failed_output_identity(self):
        rows = deepcopy(self.main)
        proof = rows[0]['cases']['fast']['warm_calls'][0]['vs_baseline']
        proof.update(raw_bits_equal=False, max_abs_error=0.125, relative_rms=0.001)
        report = aggregate(rows)
        self.assertFalse(report['same_input_mode_rank_repeat_complete_output_identity'])
        self.assertFalse(report['complete_output_identity_accepted'])
        self.assertFalse(report['official_goal_complete'])
    def test_probe_rank_order_does_not_mix_rank_wall_with_other_rank_components(self):
        forward = aggregate_components(self.probes, self.main)
        reversed_rows = aggregate_components(list(reversed(self.probes)), self.main)
        self.assertEqual(forward, reversed_rows)
        case = forward['cases']['fast']
        for vector in case['probe_vectors']:
            expected = max(vector['all_rank_vectors'], key=lambda row: row['complete_encode_ids_wall_s'])
            self.assertEqual(vector['representative'], expected)
            self.assertEqual(vector['slowest_complete_wall_rank'], expected['rank'])
    def test_probe_environment_parameter_and_input_fixture_changes_rejected(self):
        for mutation in ('environment', 'parameter_bytes', 'capture'):
            rows = deepcopy(self.probes)
            if mutation == 'environment': rows[0]['environment']['VLLM_OMNI_910A_TEXT_PAD_KV'] = '0'
            elif mutation == 'parameter_bytes': rows[0]['component_parameter_bytes']['vision'] = 0
            else: rows[0]['capture_file_sha256'] = '0' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                aggregate_components(rows, self.main)
    def test_clean_call_with_synchronizing_component_probes_cannot_be_labelled_clean(self):
        rows = deepcopy(self.probes)
        rows[0]['cases']['fast']['clean_before'][0]['parts'] = []
        with self.assertRaises(ValueError):
            aggregate_components(rows, self.main)


if __name__ == '__main__':
    unittest.main()
