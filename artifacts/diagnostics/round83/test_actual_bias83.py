"""Validate the actual portable FP32 bias kernel;micro acceptance is not model acceptance."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class ActualBiasEvidence(unittest.TestCase):
    def test_portable_bias_was_built_launched_and_repeat_verified(self):
        r=json.loads((ROOT/'results/bias01/bias_result83.json').read_text())
        self.assertTrue(r['completed']);self.assertTrue(r['kernel_launch_executed'])
        self.assertTrue(r['build']['built']);self.assertTrue(r['build']['single_final_half_cast'])
        self.assertFalse(r['build']['matmul_bias_operand_used']);self.assertTrue(r['repeat_vs_first']['raw_bits_equal'])
        self.assertEqual(hashlib.sha256((ROOT/'bias/tik_bias_epilogue83.py').read_bytes()).hexdigest(),r['kernel_source_sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'probe_bias83.py').read_bytes()).hexdigest(),r['source_sha256'])
    def test_same_CPU_Half_gold_error_reduced_without_parameter_changes(self):
        r=json.loads((ROOT/'results/bias01/bias_result83.json').read_text())
        old=r['native_vs_CPU_FP16'];new=r['kernel_vs_CPU_FP16']
        self.assertEqual(old['reference_sha256'],new['reference_sha256'])
        self.assertAlmostEqual(new['relative_RMS'],3.3662350586129384e-6)
        self.assertAlmostEqual(old['relative_RMS'],.0003517546748975196)
        self.assertGreater(old['relative_RMS']/new['relative_RMS'],100)
        self.assertTrue(r['GPU_operands_unchanged']);self.assertTrue(r['original_inputs_and_parameters_unchanged'])
        self.assertFalse(r['full_encoder_precision_accepted']);self.assertFalse(r['whole_goal_complete'])
    def test_original_native_bias_rounding_is_measured_not_assumed(self):
        r=json.loads((ROOT/'results/bias01/bias_result83.json').read_text())
        self.assertTrue(r['native_vs_Half_GEMM_then_Half_bias']['raw_bits_equal'])
        self.assertTrue(r['FP32_diagnostic_not_a_new_teacher']);self.assertTrue(r['not_H3_performance'])
        self.assertIsNone(r['numerical_acceptance_threshold'])
    def test_twoK_blocked_smoke_preserved_FP32_accumulation(self):
        r=json.loads((ROOT/'results/blocked01/blocked_smoke_result83.json').read_text())
        self.assertTrue(r['completed']);self.assertEqual(r['logical_geometry'],{'M':128,'K':256,'N':128})
        self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal'])
        self.assertLess(r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],1e-6)
        self.assertEqual(r['kernel_final_Half_vs_CPU_Half']['relative_RMS'],r['native_Half_vs_CPU_Half']['relative_RMS'])
        self.assertTrue(r['not_whole_encoder_timing']);self.assertFalse(r['full_encoder_precision_accepted'])


if __name__=='__main__':unittest.main()
