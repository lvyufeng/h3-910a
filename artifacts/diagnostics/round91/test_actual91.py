"""Actual TP8 cache identity and real full encoder clocks,not teacher timings."""
from pathlib import Path
import json
import hashlib
import statistics
import unittest
ROOT=Path(__file__).resolve().parent


class ActualFullEncoderEvidence(unittest.TestCase):
    def rows(self):
        return[json.loads((ROOT/'results/full01/ranks'/('tp8_rank%d.json'%i)).read_text())for i in range(8)]
    def test_all8_all77_boundaries_cache_bit_equal_and_native_recovered(self):
        for r in self.rows():
            self.assertTrue(r['completed']and r['native_matches_R82_output_SHA'])
            self.assertTrue(r['native_geometry_cache_all77_boundaries_bit_equal'])
            self.assertTrue(r['corrected_geometry_cache_all77_boundaries_bit_equal'])
            self.assertTrue(r['native_recovery']['raw_bits_equal'])
            self.assertTrue(r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged'])
            for c in r['cases'].values():
                self.assertEqual(len(c['raw50_layers']),50);self.assertEqual(len(c['raw27_vision_layers']),27)
                self.assertTrue(c['repeat_vs_first']['raw_bits_equal']and c['stock_vs_equivalent']['raw_bits_equal'])
                self.assertTrue(all(x['sha256_unchanged_since_prime']for x in c['geometry_snapshot']['cached'].values()))
    def test_actual_warm5paired8rank_clocks_and_output_proofs(self):
        rows=self.rows();s=json.loads((ROOT/'summary91.json').read_text())
        for name,c in s['actual_full_cases'].items():
            values=[max(r['cases'][name]['warm_stock_s'][i]for r in rows)for i in range(5)]
            self.assertEqual(values,c['paired8rank_max_real_encode_s'])
            self.assertEqual(statistics.median(values),c['paired8rank_max_median_s'])
            for r in rows:
                raw=r['cases'][name]
                self.assertEqual(len(set(raw['warm_output_SHA256'])),1)
                self.assertIn('real private encode_ids includes pixels/newvision',raw['warm_scope'])
        self.assertTrue(s['no_CPU_precomputed_image_condition_in_actual_warm'])
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
    def test_corrected_vision_reproduces_R90_without_FP32_row_cache(self):
        rows=self.rows();original=json.loads((ROOT.parent/'round90/results/vision02/vision90.json').read_text())
        expected=original['cases']['TIK_erf_patch_and_bias']['merged_vs_CPU']['npu']['sha256']
        for r in rows:
            self.assertTrue(r['FP32_row_adapter_not_installed'])
            for name in('corrected_vision_native_text','corrected_vision_native_text_geometry_cache','corrected_vision_precise_text_geometry_cache'):
                self.assertEqual(r['cases'][name]['vision_merged_vs_CPU']['npu']['sha256'],expected)
            self.assertAlmostEqual(r['CPU_condition_controls']['native_attention']['selected50_vs_CPU']['relative_rms'],.024756999879344863)
            self.assertAlmostEqual(r['CPU_condition_controls']['precise_attention']['selected50_vs_CPU']['relative_rms'],.018193357182765504)
    def test_geometry_scope19MB_and_inference_value_rereads_are_proven(self):
        for r in self.rows():
            cache=r['geometry_final']
            self.assertEqual(cache['extra_bytes'],19740672)
            self.assertTrue(cache['cached_geometry_only']);self.assertFalse(cache['caches_pixels_patch_hidden_deepstack_or_text'])
            self.assertEqual(cache['float_weights'],0)
            self.assertGreater(cache['stats']['grid_value_rereads'],0)
            self.assertTrue(all(x['sha256_unchanged_since_prime']for x in cache['cached'].values()))
    def test_executed_sources_original_refs_and_worker_cleanup(self):
        s=json.loads((ROOT/'executed_sources91.json').read_text())
        self.assertEqual(s['files']['full_encoder91.py']['sha256'],hashlib.sha256((ROOT/'full_encoder91.py').read_bytes()).hexdigest())
        self.assertEqual(s['files']['geometry_cache91.py']['sha256'],hashlib.sha256((ROOT/'cache/geometry_cache91.py').read_bytes()).hexdigest())
        self.assertEqual(len(s['kernel_objects']),12)
        v=json.loads((ROOT/'final_verification91.json').read_text())
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertEqual(v['remaining_workers'],[]);self.assertTrue(v['all_production_reference_originalHF_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
