#!/usr/bin/env python3
"""Data contracts for same-run timing, real-input gates and unadopted trim candidate."""
import ast
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/name).read_text())

class Contracts(unittest.TestCase):
    def test_current_run_retains_native_contract(self):
        summary=read('path_summary218.json')
        self.assertEqual(summary['tp'],2);self.assertEqual(summary['usp'],4)
        self.assertTrue(summary['all_155_strict_checks_per_step_per_rank'])
        self.assertTrue(summary['do_not_add_parent_child_or_copy_spans'])
        self.assertFalse(summary['candidate_installed_in_model'])
        self.assertLess(summary['profile_overhead_ratio_to_clean_pre'],1.015)
        self.assertGreater(summary['cases'][0]['rank_max_median_s'],17.5)
        self.assertEqual(summary['regions_instrumented_run_only']['FA.ACL']['calls_per_measured_step'],50)

    def test_only_invalid_padding_tasks_are_removed(self):
        plans=read('padding_plan_contracts218.json')
        self.assertTrue(plans['pass'])
        real=plans['cases'][0]
        self.assertEqual((real['Tr'],real['Tc']),(327,138))
        self.assertTrue(real['all_input_query_rows_scheduled'])
        self.assertTrue(real['all_valid_keys_scheduled'])
        self.assertTrue(real['only_fully_invalid_key_tiles_removed'])
        self.assertAlmostEqual(real['tile_work_ratio_to_padded_baseline'],0.9163011695906432)

    def test_broadcast_exact_but_slower(self):
        gate=read('broadcast_real_comparison218.json')
        self.assertEqual(gate['differing_words'],0)
        self.assertTrue(read('broadcast_unit_candidate218.json')['pass'])
        baseline=read('fa_receipts/real_base.json')
        candidate=read('fa_receipts/real_candidate.json')
        self.assertGreater(candidate['kernel_event_median_ms'],baseline['kernel_event_median_ms'])

    def test_trim_local_exact_and_faster(self):
        self.assertEqual(read('trim_comparison218.json')['differing_words'],0)
        base=read('trim_receipts/base_real.json')
        new=read('trim_receipts/trim_real.json')
        self.assertGreater(base['kernel_event_median_ms']/new['kernel_event_median_ms'],1.07)
        self.assertEqual(new['compiled_kernels'][0]['instance']['Tr'],327)
        self.assertEqual(new['compiled_kernels'][0]['instance']['Tc'],138)
        self.assertFalse(new['compiled_kernels'][0]['instance']['has_attn_mask'])
        for row in read('boundary_trim218.json')['checks']:
            self.assertEqual(row['differing_words'],0)
            self.assertEqual(row['nonfinite'],0)
            self.assertEqual(row['saturated'],0)

    def test_real_input_gate_and_single_candidate(self):
        summary=read('trim_model_summary218.json')
        self.assertEqual(summary['total_gate_calls'],400)
        self.assertEqual(summary['total_gate_elements'],14978252800)
        self.assertEqual(summary['total_different_words'],0)
        self.assertTrue(summary['all50blocks_and_155strict_checks_retained'])
        self.assertTrue(summary['no_other_candidate_combined'])
        self.assertTrue(summary['nothing_adopted'])
        self.assertFalse(summary['whole_goal_complete'])
        self.assertFalse(summary['candidate_default_enabled'])
        self.assertGreater(summary['measured_rank_max_step_gain_s'],0.5)
        self.assertLess(summary['measured_rank_max_step_gain_s'],0.7)
        self.assertLess(summary['restore_time_ratio_to_pre'],1.01)
        for gate in summary['gate']:
            self.assertEqual(gate['different_words'],0)
            self.assertTrue(gate['restored'])

    def test_model_repeat_scatter_not_promoted(self):
        metrics=read('trim_prediction_metrics218.json')
        self.assertTrue(metrics['native_repeat_is_not_a_tolerance'])
        self.assertTrue(metrics['no_full_model_numeric_promotion'])
        self.assertGreater(metrics['comparisons'][0]['metrics']['video_velocity']['different_words'],0)
        self.assertIn('Awaiting',read('trim_model_summary218.json')['decision'])

    def test_cleanup_verified(self):
        record=read('cleanup218.json')
        self.assertTrue(record['source_unchanged'])
        self.assertTrue(record['all_host_only_tensor_outputs_removed'])
        self.assertTrue(record['private_candidate_binary_caches_removed'])
        self.assertTrue(record['no_device_driver_or_sdk_reset'])
        self.assertFalse(record['private_model_or_FA_processes'])
        self.assertIn('released=true',record['lease_release'])

    def test_our_sources_parse(self):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

if __name__=='__main__':unittest.main()
