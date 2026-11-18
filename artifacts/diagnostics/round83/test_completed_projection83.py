"""Validate completed H3-sized primitive cases without overstating final-Half accuracy."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class CompletedProjectionEvidence(unittest.TestCase):
    def test_KV_and_Q_really_executed_40_FP32_K_blocks(self):
        for case,n,cores in(('kv',128,16),('q',1024,30)):
            r=json.loads((ROOT/'results/blocked02'/('blocked_'+case+'_result83.json')).read_text())
            self.assertTrue(r['completed']);self.assertTrue(r['kernel_launch_executed'])
            self.assertEqual(r['logical_geometry'],{'M':2043,'K':5120,'N':n})
            self.assertEqual(r['build']['block_dim'],cores);self.assertEqual(r['build']['dst_dtype'],'float32')
            self.assertEqual(r['build']['accumulation']['k_tiles'],40)
            self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal']);self.assertEqual(r['padded_output_rows_max_abs'],0)
            self.assertLess(r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],2e-6)
            self.assertEqual(hashlib.sha256((ROOT/'blocked/blocked_fp32_gemm83.py').read_bytes()).hexdigest(),r['kernel_source_sha256'])
    def test_small_KV_native_nondeterminism_not_hidden_or_mislabelled_accuracy_gain(self):
        r=json.loads((ROOT/'results/blocked02/blocked_kv_result83.json').read_text())
        self.assertFalse(r['native_repeat_vs_first']['raw_bits_equal'])
        self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal'])
        self.assertGreater(r['kernel_final_Half_vs_CPU_Half']['relative_RMS'],r['native_Half_vs_CPU_Half']['relative_RMS'])
        self.assertFalse(r['full_encoder_precision_accepted'])
    def test_FP32_output_layout_cost_not_reported_as_same_dtype_speedup(self):
        for case in('kv','q'):
            r=json.loads((ROOT/'results/blocked02'/('blocked_'+case+'_result83.json')).read_text())
            self.assertTrue(r['native_returns_Half_while_custom_retains_FP32'])
            self.assertGreater(r['medians_s']['resident_with_layouts'],r['medians_s']['native_Half'])
            self.assertTrue(r['not_whole_encoder_timing']);self.assertIsNone(r['precision_acceptance_threshold_chosen'])
            self.assertFalse(r['whole_goal_complete'])


if __name__=='__main__':unittest.main()
