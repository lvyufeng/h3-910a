"""Actual BC256 compile, poisoned-pad proof and the negative real-QKV speed result."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class Actual103Evidence(unittest.TestCase):
    def test_real_smoke_original_CPUHalf_constant_and_general_float_accuracy(self):
        r=json.loads((ROOT/'results/smoke01/bc256_smoke103.json').read_text());self.assertTrue(r['completed'])
        self.assertEqual(r['meta']['block_dim'],4)
        self.assertEqual(r['meta']['source_sha256'],hashlib.sha256((ROOT/'flash_bc256/flash_vision103.py').read_bytes()).hexdigest())
        for case in r['cases'].values():
            self.assertTrue(case['poison_vs_zero_pad']['raw_bits_equal'])
            for c in case['padding'].values():self.assertTrue(c['all_repeats_and_inputs_unchanged'])
        self.assertTrue(r['cases']['constant_uniform']['padding']['zero']['vs_CPU_Half_MATH']['raw_bits_equal'])
        self.assertTrue(r['cases']['constant_uniform']['padding']['poison']['vs_CPU_Half_MATH']['raw_bits_equal'])
    def test_actual_realQKV_has_oldBC128_SHA_shape_repeat_and_cpu_oracle_gates(self):
        r=json.loads((ROOT/'results/leaf01/leaf_compare103.json').read_text());self.assertTrue(r['completed'])
        expected=json.loads((ROOT/'expected_inputs103.json').read_text())
        for image,c in enumerate(r['images']):
            self.assertTrue(c['all_warm_repeats_and_QKV_unchanged'])
            self.assertEqual(c['old_vs_CPU_Half_MATH']['npu']['sha256'],expected['R101_BC128_leaf_output_SHAs'][image])
            self.assertEqual(c['flash_vs_CPU_Half_MATH']['npu']['shape'],[1,16,4032,72])
            self.assertEqual(len(c['old_warm_s']),7)
            self.assertLess(c['old_over_flash_median_speedup'],1.0)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance'])


if __name__=='__main__':unittest.main()
