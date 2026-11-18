"""Actual full-graph tests explicitly retain the combined-path regression."""
from pathlib import Path
import hashlib
import json
import unittest
ROOT=Path(__file__).resolve().parent


class FullAttentionEvidence(unittest.TestCase):
    def test_all8_full_graphs_repeated_and_native_recovered(self):
        for rank in range(8):
            r=json.loads((ROOT/'results/full01/ranks'/('tp8_rank%d.json'%rank)).read_text())
            self.assertTrue(r['completed']and r['frozen_inputs_unchanged'])
            self.assertTrue(r['native_matches_round81_sha256']and r['native_after_row_AB']['raw_bits_equal'])
            self.assertEqual(r['source_sha256'],hashlib.sha256((ROOT/'full_attention85.py').read_bytes()).hexdigest())
            for case in r['row_full_cases'].values():
                self.assertEqual(len(case['raw_layers']),50);self.assertTrue(case['repeat_vs_first']['raw_bits_equal'])
            e=r['accurate_attention_final'];self.assertEqual(e['calls'],502)
            self.assertEqual(e['kernel_calls'],6024);self.assertEqual(e['half_output_chunks'],2008)
            self.assertEqual(e['actual_FP32_before_Half_chunks'],2008)
            self.assertTrue(r['cleanup']['accurate_attention_uninstalled'])
    def test_better_leaf_does_not_guarantee_better_combined_graph(self):
        s=json.loads((ROOT/'summary85.json').read_text());c=s['full50_cases']
        self.assertAlmostEqual(c['native_rows_CPU_condition']['relative_RMS'],.024756999879344863)
        self.assertAlmostEqual(c['accurate_attention_native_rows_CPU_condition']['relative_RMS'],.018193357182765504)
        self.assertGreater(c['accurate_attention_FP32_rows_CPU_condition']['relative_RMS'],c['native_rows_CPU_condition']['relative_RMS'])
        self.assertAlmostEqual(c['accurate_attention_FP32_rows_native_condition']['relative_RMS'],.07609362545527779)
        self.assertGreater(s['full_encode_rank_max_median_s'],3)
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
    def test_observed_error_growth_is_late_not_hidden(self):
        d=json.loads((ROOT/'layer_drift85.json').read_text())
        row=d['accurate_attention_FP32_rows_CPU_condition']
        self.assertLess(row[0]['relative_RMS'],8e-5)
        self.assertLess(next(x for x in row if x['layer']==24)['relative_RMS'],5e-4)
        self.assertGreater(next(x for x in row if x['layer']==49)['relative_RMS'],.02)
    def test_worker_and_production_cleanup_verified(self):
        v=json.loads((ROOT/'final_verification85.json').read_text())
        self.assertFalse(v['remaining_workers']);self.assertEqual(len(v['production']),23)
        self.assertEqual(len(v['installed_api_sources']),13)
        self.assertTrue(v['all_production_reference_and_launcher_unchanged'])


if __name__=='__main__':unittest.main()
