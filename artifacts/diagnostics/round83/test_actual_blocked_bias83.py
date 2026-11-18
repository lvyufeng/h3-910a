"""Actual H3-sized portable FP32 bias epilogue evidence,not full-encoder acceptance."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class ActualBlockedBias(unittest.TestCase):
    def test_both_smoke_and_H3_bias_cases_repeated_with_original_parameters(self):
        for case in('smoke','kv'):
            p=ROOT/'results/blocked_bias01'/('blocked_bias_'+case+'_result83.json');r=json.loads(p.read_text())
            self.assertTrue(r['completed']);self.assertTrue(r['kernel_launch_executed'])
            self.assertTrue(r['repeat_vs_first']['raw_bits_equal']);self.assertTrue(r['GPU_operands_unchanged'])
            self.assertTrue(r['original_inputs_and_parameters_unchanged']);self.assertFalse(r['build']['matmul_bias_operand_used'])
            self.assertTrue(r['build']['single_final_half_cast'])
            self.assertEqual(hashlib.sha256((ROOT/'blocked_bias_epilogue83.py').read_bytes()).hexdigest(),r['kernel_source_sha256'])
            self.assertEqual(hashlib.sha256((ROOT/'probe_blocked_bias83.py').read_bytes()).hexdigest(),r['source_sha256'])
    def test_H3_size_bias_error_is_reduced_against_the_same_CPUHalf_gold(self):
        r=json.loads((ROOT/'results/blocked_bias01/blocked_bias_kv_result83.json').read_text())
        self.assertEqual(r['geometry'],{'M':2043,'K':5120,'N':128})
        self.assertAlmostEqual(r['kernel_vs_CPU_FP16']['relative_RMS'],1.3543982840947204e-5)
        self.assertAlmostEqual(r['native_vs_CPU_FP16']['relative_RMS'],.00035228753704752826)
        self.assertGreater(r['native_vs_CPU_FP16']['relative_RMS']/r['kernel_vs_CPU_FP16']['relative_RMS'],25)
        self.assertEqual(r['kernel_vs_CPU_FP16']['reference_sha256'],r['native_vs_CPU_FP16']['reference_sha256'])
        self.assertFalse(r['full_encoder_precision_accepted']);self.assertTrue(r['not_H3_performance'])
    def test_summary_does_not_substitute_primitives_for_model_acceptance(self):
        s=json.loads((ROOT/'summary83.json').read_text())
        self.assertTrue(s['all_four_H3_projection_shapes_executed'])
        self.assertEqual(set(s['H3_projection_cases']),{'kv','q','up','row'})
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
        self.assertTrue(s['no_whole_encoder_speedup_from_primitive_timing'])


if __name__=='__main__':unittest.main()
