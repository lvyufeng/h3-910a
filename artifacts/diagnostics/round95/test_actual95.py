"""Actual full TP8 CPU-exact RoPE intervention keeps all non-monotonic results."""
from pathlib import Path
import json
import hashlib
import statistics
import unittest
ROOT=Path(__file__).resolve().parent


class ActualFullEncoderEvidence(unittest.TestCase):
    def rows(self):return[json.loads((ROOT/'results/full01/ranks'/('tp8_rank%d.json'%i)).read_text())for i in range(8)]
    def test_all8sixcases_repeats_stock_equivalence_native_recovery(self):
        for r in self.rows():
            self.assertTrue(r['completed']and r['native_matches_R82_output_SHA'])
            self.assertTrue(r['native_recovery']['raw_bits_equal']and r['parameter_and_fixture_SHA_unchanged']and r['CPU_reference_SHA_unchanged'])
            self.assertEqual(len(r['cases']),6)
            for c in r['cases'].values():
                self.assertEqual(len(c['raw50_layers']),50);self.assertEqual(len(c['raw27_vision_layers']),27)
                self.assertTrue(c['repeat_vs_first']['raw_bits_equal']and c['stock_vs_equivalent']['raw_bits_equal'])
                self.assertEqual(len(set(c['warm_output_SHA256'])),1)
    def test_original_CPU_tables_1MB_and_vision_all27bits_unchanged(self):
        for r in self.rows():
            prime=r['CPU_rope_prime'];self.assertTrue(prime['cpu_table']['repeat_bits_equal']and prime['cpu_table']['matches_R94_measured_CPU_table'])
            self.assertTrue(prime['native_probe']['matches_R94_observed_native_SHA'])
            self.assertEqual(prime['teacher_weights'],0);self.assertEqual(prime['extra_bytes'],1046016)
            self.assertTrue(all(r['vision_boundaries_unchanged_by_CPU_rope'].values()))
            for entry in r['CPU_rope_final']['cached'].values():self.assertTrue(entry['sha256_unchanged_since_prime'])
    def test_all_five_paired8max_clocks_exact_and_no_sub2_claim(self):
        s=json.loads((ROOT/'summary95.json').read_text());rows=self.rows()
        for name,c in s['cases'].items():
            clocks=[max(r['cases'][name]['warm_stock_s'][i]for r in rows)for i in range(5)]
            self.assertEqual(clocks,c['paired8rankmax_warm_s']);self.assertEqual(statistics.median(clocks),c['paired8rankmax_median_s'])
            self.assertFalse(c['all5paired_max_under2s'])
        self.assertFalse(s['whole_goal_complete']or s['full_encoder_precision_accepted']);self.assertIsNone(s['numerical_tolerance'])
    def test_nonmonotonic_full_precision_and_CPUcondition_negatives_preserved(self):
        s=json.loads((ROOT/'summary95.json').read_text());c=s['cases'];controls=s['CPU_condition_accuracy_only']
        self.assertGreater(c['native_CPU_rope']['selected50_CPU_relative_RMS'],c['native']['selected50_CPU_relative_RMS'])
        self.assertLess(c['corrected_vision_native_text_CPU_rope']['selected50_CPU_relative_RMS'],c['corrected_vision_native_text']['selected50_CPU_relative_RMS'])
        self.assertGreater(c['corrected_vision_precise_text_CPU_rope']['selected50_CPU_relative_RMS'],c['corrected_vision_precise_text']['selected50_CPU_relative_RMS'])
        self.assertGreater(controls['native_attention_CPU_rope']['relative_RMS'],controls['native_attention']['relative_RMS'])
        self.assertGreater(controls['precise_attention_CPU_rope']['relative_RMS'],controls['precise_attention']['relative_RMS'])
        self.assertTrue(all(x['not_full_encoder_warmtiming']for x in controls.values()))
    def test_all_executed_sources_12kernels_and_final_cleanup(self):
        s=json.loads((ROOT/'executed_sources95.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'full_encoder95.py').read_bytes()).hexdigest(),s['sources']['full_encoder95.py']['sha256'])
        self.assertEqual(hashlib.sha256((ROOT/'cache/cpu_text_rope_cache95.py').read_bytes()).hexdigest(),s['sources']['cpu_text_rope_cache95.py']['sha256'])
        self.assertEqual(len(s['kernels']),12)
        v=json.loads((ROOT/'final_verification95.json').read_text());self.assertEqual(v['remaining_workers'],[])
        self.assertEqual(len(v['production']),23);self.assertEqual(len(v['installed_api_sources']),13)
        self.assertTrue(v['all_production_reference_originalHF_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
