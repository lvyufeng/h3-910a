"""Static and actual evidence contracts for FP16 operator factor experiments."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class FP16OperatorEvidence(unittest.TestCase):
    def test_leaf_sources_match_actual_receipts(self):
        for source,receipt in(('leaf_ops80.py','operators.json'),('leaf_factors80.py','factors.json'),('precision_policy80.py','policy.json'),('pv_split80.py','pv_split.json'),('vision_factors80.py','vision_factors.json'),('vision_compensated80.py','vision_comp.json')):
            r=json.loads((ROOT/receipt).read_text());self.assertTrue(r['completed'])
            self.assertEqual(hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),r['source_sha256'])
            self.assertFalse(r['production_changed']);self.assertFalse(r['whole_goal_complete'])
    def test_biased_linear_extra_round_is_measured_not_inferred(self):
        r=json.loads((ROOT/'factors.json').read_text());c=r['cases']
        self.assertTrue(c['Linear_native_with_bias_vs_NPU_two_rounding_factor']['raw_bits_equal'])
        self.assertGreater(c['Linear_with_bias_native_vs_CPU']['relative_RMS'],30*c['Linear_no_bias_native_vs_CPU']['relative_RMS'])
        self.assertFalse(r['grouped_matmul_FP16_inputs_FP32_output']['supported'])
        self.assertIn('y dtype should be FLOAT16',r['grouped_matmul_FP16_inputs_FP32_output']['failure']['message'])
    def test_layernorm_classification_prevents_false_bulk_12percent_claim(self):
        c=json.loads((ROOT/'factors.json').read_text())['cases']
        self.assertGreater(c['LayerNorm_native_constant']['max_abs_error'],.9)
        self.assertTrue(c['LayerNorm_stable_constant']['raw_bits_equal'])
        self.assertLess(c['LayerNorm_native_ordinary']['relative_RMS'],1e-5)
        self.assertFalse(c['LayerNorm_stable_repeat']['raw_bits_equal'])
    def test_hf32_flag_hypothesis_is_rejected_on_this_actual_probe(self):
        r=json.loads((ROOT/'policy.json').read_text());pv=r['cases']['Attention_PV']
        self.assertEqual(r['policy_before'],r['policy_restored'])
        self.assertFalse(r['policy_before']['matmul_allow_hf32'])
        self.assertEqual(pv['default']['NPU_FP32_vs_CPU_FP32'],pv['false']['NPU_FP32_vs_CPU_FP32'])
        self.assertEqual(pv['false']['NPU_FP32_vs_CPU_FP32'],pv['true']['NPU_FP32_vs_CPU_FP32'])
        self.assertFalse(r['CPU_reference_redefined'])
    def test_split_probability_is_not_claimed_exact_or_free(self):
        c=json.loads((ROOT/'pv_split.json').read_text())['cases']
        self.assertLess(c['split_scale_65536']['same_input_CPU_FP32']['relative_RMS'],1e-6)
        self.assertGreater(c['split_scale_65536']['FP32_probability_reconstruction']['relative_RMS'],0)
        r=json.loads((ROOT/'vision_comp.json').read_text());p=r['phases']
        self.assertLess(p['fp32_bias_compensated_pv']['CPU_FP16_comparison']['merged']['relative_RMS'],p['native_before']['CPU_FP16_comparison']['merged']['relative_RMS'])
        self.assertGreater(p['fp32_bias_compensated_pv']['calls'][1]['resident_vision_s'],2)
        self.assertTrue(r['native_restoration']['merged']['raw_bits_equal'])
        self.assertEqual(r['parameters_before'],r['parameters_after']);self.assertTrue(r['input_unchanged'])
    def test_small_N128_nondeterminism_and_zero_padding_controls(self):
        r=json.loads((ROOT/'kv_padding.json').read_text());self.assertTrue(r['completed'])
        self.assertEqual(hashlib.sha256((ROOT/'kv_padding80.py').read_bytes()).hexdigest(),r['source_sha256'])
        for item in r['cases']:
            self.assertTrue(item['extra_rows_all_zero']);self.assertTrue(item['parameter_prefix_unchanged'])
            self.assertEqual(len(item['repeat']),8)
            if item['padded_output_rows']==128:self.assertFalse(any(p['raw_bits_equal']for p in item['repeat']))
            else:self.assertTrue(all(p['raw_bits_equal']for p in item['repeat']))
        o=json.loads((ROOT/'offset.json').read_text());self.assertFalse(o['cases'][0]['clone_repeat']['raw_bits_equal'])
    def test_independent_CPU_gold_really_completed(self):
        r=json.loads((ROOT.parent/'round78'/'reference.json').read_text());v=json.loads((ROOT.parent/'round78'/'final_CPU_reference_verification78.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['full_multimodal_reference_completed'])
        self.assertTrue(r['repeat_pre_final_norm_bits_equal']);self.assertTrue(r['repeat_post_final_norm_bits_equal'])
        self.assertEqual(len(r['vision_layers']),27);self.assertEqual(len(r['language_layers']),50)
        self.assertTrue(v['CPU_FP16_reference_verified']);self.assertFalse(v['NPU_precision_accepted']);self.assertFalse(v['whole_goal_complete'])


if __name__=='__main__':unittest.main()
