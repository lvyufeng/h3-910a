"""Eight actual BC128 encoder ranks, frozen oracle and genuine untimed output capture."""
from pathlib import Path
import json
import unittest
import collect102
ROOT=Path(__file__).resolve().parent


class Actual102Evidence(unittest.TestCase):
    def test_full50_full27_R101_shapes_repeats_HCCL_equivalence_recovery(self):
        r=collect102.collect(ROOT/'results/full01')
        self.assertTrue(r['completed']and r['all8_native_recovery_parameters_fixture_originalCPUgold_unchanged'])
        for case in r['cases'].values():
            self.assertTrue(case['all8_selected50_and_all50_text_boundary_SHAs_equal'])
            self.assertTrue(case['all8_repeat_stock_equivalent_and_R101_visual_SHA_verified'])
            self.assertEqual(case['selected50_relativeRMS_all8_min_max'][0],case['selected50_relativeRMS_all8_min_max'][1])
            self.assertEqual(len(case['paired_8rankmax_clock_s']),5)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance']);self.assertTrue(r['no_default_promotion'])
    def test_summary_recomputes_from_all8_raw_samples(self):
        self.assertEqual(json.loads((ROOT/'summary102.json').read_text()),collect102.collect(ROOT/'results/full01'))
    def test_genuine_rank0_outputs_saved_separately_and_never_other_rank_capture(self):
        for rank in range(8):
            r=json.loads((ROOT/('results/full01/tp8_rank%d.json'%rank)).read_text())
            for name,case in r['cases'].items():
                if rank==0:
                    actual=case['actual_NPU_output_saved_outside_timer']
                    self.assertTrue(actual['file_SHA256']);self.assertIn('NPU_selected50_'+name+'.pt',actual['path'])
                else:self.assertNotIn('actual_NPU_output_saved_outside_timer',case)
    def test_CPU_only_audit_all23production13API_originalgold_objects_and_workers(self):
        r=json.loads((ROOT/'final_source_verification_cpu102.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertTrue(r['all_relevant_workers_zero']);self.assertFalse(r['remaining_workers'])
        self.assertTrue(r['all8_R102_encoder_results_complete_native_recovery_verified'])
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged'])
        self.assertEqual(len(r['actual_saved_outputs']),4)
        for row in r['actual_saved_outputs']:
            self.assertTrue(row['captured_outside_warm_clock_not_used_as_input']);self.assertEqual(row['dtype'],'Half')
            self.assertTrue(row['file_SHA256']and row['tensor_SHA256'])


if __name__=='__main__':unittest.main()
