#!/usr/bin/env python3
"""Contracts prevent promotion of diagnostics/fit hypotheses to measured performance claims."""
import json
import re
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DOC=ROOT/'PERFORMANCE_STATUS.md'

class Tests(unittest.TestCase):
    def test_current_target_and_evidence(self):
        text=DOC.read_text()
        m=re.search(r'截至 R(\d+)',text)
        self.assertIsNotNone(m)
        self.assertGreaterEqual(int(m.group(1)),217)
        self.assertIn('≤10 s/step',text)
        self.assertIn('上调',text)
        self.assertIn('总体目标未完成',text)
        self.assertIn('17.987→15.473',text)
        self.assertIn('2.514s',text)

    def test_no_promotion_of_unapproved_or_imagined_results(self):
        text=DOC.read_text()
        self.assertIn('不合规诊断',text)
        self.assertIn('不是已实现加速',text)
        self.assertIn('默认保留',text)
        self.assertIn('不能据此宣称可达/不可达',text)
        self.assertIn('INCONCLUSIVE',text)
        self.assertIn('0.5s预测未实测',text)

    def test_dense_work_is_supported_by_actual_build(self):
        summary=json.loads((ROOT/'round217/summary217.json').read_text())
        geometry=summary['geometry']
        self.assertFalse(geometry['has_attn_mask'])
        self.assertFalse(geometry['only_causal'])
        self.assertEqual(geometry['next_block_num'],65536)
        self.assertEqual(summary['work']['dense_tiles_per_head'],342*144)
        self.assertEqual(summary['work']['dense_tiles_all_heads'],342*144*7)
        self.assertAlmostEqual(summary['work']['padded_dense_matmul_TFLOP_per_call'],6.868156022784)
        self.assertTrue(summary['work']['empirical_GEMM_reference_is_not_a_physical_ceiling'])

    def test_corrected_candidates_are_exact_but_rejected(self):
        summary=json.loads((ROOT/'round217/summary217.json').read_text())
        self.assertTrue(summary['nothing_adopted'])
        self.assertFalse(summary['whole_goal_complete'])
        self.assertTrue(summary['grouped_double_and_single_binary_equal'])
        for row in summary['measurements'][1:3]:
            self.assertGreater(row['kernel_time_ratio_to_baseline'],1.01)
            self.assertTrue(row['output_sha256_matches_baseline'])
            self.assertIn('Reject',row['disposition'])
        for row in summary['numeric']['comparisons']:
            self.assertEqual(row['differing_words'],0)
            self.assertEqual(row['candidate_nonfinite'],0)
            self.assertEqual(row['candidate_saturated'],0)

    def test_failed_layout_not_rounding_tolerance(self):
        summary=json.loads((ROOT/'round217/summary217.json').read_text())
        self.assertGreater(summary['failed_direct_PV_layout']['relative_rms'],1.0)
        self.assertFalse(summary['failed_direct_PV_layout']['candidate_enabled'])
        self.assertIn('507015',summary['scoped_sync_runtime_fault'])

    def test_scenario_arithmetic(self):
        s=json.loads((ROOT/'round217/summary217.json').read_text())['conditional_scenarios_only']
        self.assertTrue(s['no_combo_measured'])
        self.assertTrue(s['different_run_scopes'])
        self.assertAlmostEqual(s['FA_speedup_1_5_assuming_guard_step_s'],11.3233333333)
        self.assertAlmostEqual(s['FA_speedup_1_89_assuming_guard_step_s'],10.2979153439)
        self.assertAlmostEqual(s['FA_speedup_needed_with_guard_assumption'],2.0444322545)

    def test_cleanup_and_timer(self):
        receipt=json.loads((ROOT/'round217/cleanup217.json').read_text())
        self.assertTrue(receipt['baseline_source_unchanged'])
        self.assertTrue(receipt['no_device_or_driver_reset'])
        self.assertTrue(receipt['all_tensor_outputs_removed_after_scalar_comparison'])
        self.assertIn('released=true',receipt['lease_release'])
        restored=json.loads((ROOT/'round217/restored_end217.json').read_text())
        self.assertIn('native ACL',restored['kernel_timing_method'])
        self.assertGreater(restored['kernel_event_median_ms'],150)
        self.assertLess(restored['kernel_event_median_ms'],156)

if __name__=='__main__':
    unittest.main()
