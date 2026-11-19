"""Grid-aware audit regression: legitimate smallTP GEMMs use5/9cores, not all30."""
from pathlib import Path
import json
import unittest
ROOT=Path(__file__).resolve().parent


class Audit108GridContracts(unittest.TestCase):
    def test_actual_small_GEMM_launch_counts_match_min_tasks_and_original_30limit(self):
        r=json.loads((ROOT.parent/'round107/results/gemm_tp01/prebuild107.json').read_text())
        self.assertTrue(r['completed'])
        actual=set()
        for item in r['kernels']:
            meta=item['meta'];grid=meta['grid'];self.assertEqual(meta['block_dim'],grid['cores'])
            self.assertEqual(grid['cores'],min(30,grid['tasks']));actual.add(meta['block_dim'])
        self.assertEqual(actual,{4,5,9,30})
    def test_corrected_source_and_all_saved_outputs_verified_not_numeric_waiver(self):
        r=json.loads((ROOT/'final_source_verification_grid_fixed108.json').read_text())
        self.assertEqual(len(r['production']),23);self.assertEqual(len(r['installed_api_sources']),13)
        self.assertEqual(len(r['execution_sources']),60);self.assertEqual(len(r['kernel_objects']),34)
        self.assertTrue(r['all_relevant_workers_zero']);self.assertFalse(r['remaining_workers'])
        self.assertTrue(r['all_production_API_execution_source_originalHF_CPUgold_frozenQKV_launcher_and_recorded_binaries_unchanged'])
        self.assertTrue(r['all8_R108_encoder_results_complete_native_recovery_verified']);self.assertEqual(len(r['actual_saved_outputs']),5)
        for output in r['actual_saved_outputs']:self.assertTrue(output['captured_outside_warm_clock_not_used_as_input'])


if __name__=='__main__':unittest.main()
