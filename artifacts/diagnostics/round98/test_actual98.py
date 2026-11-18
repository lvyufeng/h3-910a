"""Completed whole TP8 experiment and read-only original-source/binary audit."""
from pathlib import Path
import json
import unittest
import collect98
ROOT=Path(__file__).resolve().parent


class ActualFull98Evidence(unittest.TestCase):
    def test_all8_complete_equal_final_all50_boundaries_and_native_recovery(self):
        r=collect98.collect(ROOT/'results/full02');self.assertTrue(r['completed'])
        self.assertTrue(r['all8_native_recovery_parameters_fixture_originalCPUgold_unchanged'])
        for case in r['cases'].values():
            self.assertTrue(case['all8_selected50_and_all50_text_boundary_SHAs_equal'])
            self.assertTrue(case['all8_repeat_stock_equivalent_and_R97_visual_SHA_verified'])
            self.assertEqual(case['selected50_relativeRMS_all8_min_max'][0],case['selected50_relativeRMS_all8_min_max'][1])
            self.assertEqual(len(case['paired_8rankmax_clock_s']),5)
        self.assertTrue(r['all_full_stock_samples_at_least2s'])
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance']);self.assertTrue(r['no_default_promotion'])
    def test_saved_summary_recomputes_exactly_not_single_rank_clock(self):
        expected=json.loads((ROOT/'summary98.json').read_text())
        self.assertEqual(expected,collect98.collect(ROOT/'results/full02'))
        self.assertIn('paired_8rankmax_clock_s',expected['cases']['native'])
    def test_18_binaries_23production_13API_sources_and_old_gold_unchanged(self):
        r=json.loads((ROOT/'final_source_verification98.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertEqual(len(r['kernel_objects']),18);self.assertFalse(r['remaining_workers'])
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_and_launcher_unchanged'])
        self.assertTrue(r['all8encoder_results_complete_native_recovery_verified'])
        self.assertTrue(all(x['unchanged']for x in r['execution_sources'].values()))
    def test_softmax_meta_alias_recording_defect_is_not_zero_kernels(self):
        r=json.loads((ROOT/'results/full02/tp8_rank0.json').read_text())
        self.assertEqual(r['vision_attention_prepare']['softmax_pack_metas'],{})
        self.assertEqual(r['cases']['corrected_NZ_vision_native_text']['vision_attention_snapshot']['softmax_kernels'],2)
        prebuild=json.loads((ROOT/'results/full02/prebuild98.json').read_text())
        self.assertEqual(len(prebuild['vision_softmax']),2)
        self.assertTrue(all(x['obj_sha256']for x in prebuild['vision_softmax']))


if __name__=='__main__':unittest.main()
