"""Validate complete H3-shaped FP32 kernels and the static/compact output match."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class DynamicProjectionEvidence(unittest.TestCase):
    def test_compact_loop_built_and_ran_q_up_row(self):
        for case,k,n in(('q',5120,1024),('up',5120,3456),('row',3456,5120)):
            p=ROOT/'results/blocked_dynamic01'/('blocked_'+case+'_result83.json');r=json.loads(p.read_text())
            self.assertTrue(r['completed']);self.assertTrue(r['kernel_launch_executed'])
            self.assertEqual(r['logical_geometry'],{'M':2043,'K':k,'N':n})
            self.assertEqual(r['build']['block_dim'],30);self.assertEqual(r['build']['dst_dtype'],'float32')
            self.assertTrue(r['kernel_repeat_vs_first']['raw_bits_equal']);self.assertEqual(r['padded_output_rows_max_abs'],0)
            self.assertLess(r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS'],2e-6)
            self.assertEqual(hashlib.sha256((ROOT/'blocked_dynamic_steps83.py').read_bytes()).hexdigest(),r['kernel_source_sha256'])
            self.assertEqual(hashlib.sha256((ROOT/'probe_blocked_dynamic83.py').read_bytes()).hexdigest(),r['source_sha256'])
    def test_compact_and_static_Q_are_bit_identical_and_same_input(self):
        static=json.loads((ROOT/'results/blocked02/blocked_q_result83.json').read_text())
        compact=json.loads((ROOT/'results/blocked_dynamic01/blocked_q_result83.json').read_text())
        self.assertEqual(static['kernel_FP32_vs_accumulator_diagnostic']['sha256'],compact['kernel_FP32_vs_accumulator_diagnostic']['sha256'])
        self.assertEqual(static['kernel_FP32_vs_accumulator_diagnostic']['reference_sha256'],compact['kernel_FP32_vs_accumulator_diagnostic']['reference_sha256'])
    def test_cost_and_scope_prevent_a_false_speedup_claim(self):
        for case in('q','up','row'):
            r=json.loads((ROOT/'results/blocked_dynamic01'/('blocked_'+case+'_result83.json')).read_text())
            self.assertGreater(r['medians_s']['resident_with_layouts'],r['medians_s']['native_Half'])
            self.assertTrue(r['native_returns_Half_while_custom_retains_FP32'])
            self.assertFalse(r['full_encoder_precision_accepted']);self.assertFalse(r['whole_goal_complete'])
            self.assertTrue(r['not_whole_encoder_timing'])
    def test_failed_static_up_was_not_silently_overwritten_or_replayed(self):
        r=json.loads((ROOT/'results/blocked02/run_finished.json').read_text());self.assertEqual(r['exit_code'],124)
        self.assertFalse((ROOT/'results/blocked02/blocked_up_result83.json').exists())
        done=json.loads((ROOT/'results/blocked_dynamic01/run_finished.json').read_text());self.assertEqual(done['exit_code'],0)


if __name__=='__main__':unittest.main()
