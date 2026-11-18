"""Actual completed full-streaming TP8 metrics and immutable source/object proof."""
from pathlib import Path
import json
import unittest
import collect100
ROOT=Path(__file__).resolve().parent


class Actual100Evidence(unittest.TestCase):
    def test_all8_complete_full50_boundary_equivalence_and_native_recovery(self):
        r=collect100.collect(ROOT/'results/full01')
        self.assertTrue(r['completed']);self.assertTrue(r['all8_native_recovery_parameters_fixture_originalCPUgold_unchanged'])
        for case in r['cases'].values():
            self.assertTrue(case['all8_selected50_and_all50_text_boundary_SHAs_equal'])
            self.assertTrue(case['all8_repeat_stock_equivalent_and_R99_visual_SHA_verified'])
            self.assertEqual(case['selected50_relativeRMS_all8_min_max'][0],case['selected50_relativeRMS_all8_min_max'][1])
            self.assertEqual(len(case['paired_8rankmax_clock_s']),5)
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertTrue(r['no_default_promotion']and r['all_full_stock_samples_at_least2s'])
    def test_saved_summary_is_recomputable_from_all_eight_ranks(self):
        self.assertEqual(json.loads((ROOT/'summary100.json').read_text()),collect100.collect(ROOT/'results/full01'))
    def test_all_recorded_sources_objects_originalgold_and_workers_audited(self):
        r=json.loads((ROOT/'final_source_verification100.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertEqual(len(r['execution_sources']),89);self.assertEqual(len(r['kernel_objects']),23)
        self.assertFalse(r['remaining_workers']);self.assertTrue(r['all_relevant_workers_zero'])
        self.assertTrue(r['all8_R100_encoder_results_complete_native_recovery_verified'])
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged'])
        self.assertTrue(all(x['unchanged']for x in r['execution_sources'].values()))
    def test_actual_helper_metadata_retained_after_uninstall(self):
        r=json.loads((ROOT/'results/full01/tp8_rank0.json').read_text())
        meta=r['vision_attention_prepare']['kernel']
        self.assertTrue(meta['obj_sha256']);self.assertEqual(meta['GM_argument_count'],5)
        self.assertEqual(meta['block_dim'],30)


if __name__=='__main__':unittest.main()
