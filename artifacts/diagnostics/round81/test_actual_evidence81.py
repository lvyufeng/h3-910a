"""Validate actual complete TP8 KV-padded repeat/timing/precision evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import unittest

ROOT=Path(__file__).resolve().parent


class ActualKVFullEvidence(unittest.TestCase):
    def test_all8_complete_repeated_encoder_outputs_and_fixed_weights(self):
        summary=json.loads((ROOT/'summary81.json').read_text())
        self.assertEqual(len(summary['receipts']),8)
        for item in summary['receipts']:
            path=ROOT/'ranks'/('tp8_rank%d.json'%item['rank']);r=json.loads(path.read_text())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['sha256'])
            self.assertTrue(r['completed']);self.assertTrue(r['cases']['repeat_vs_first']['raw_bits_equal'])
            self.assertEqual(len(r['language_layers']),50);self.assertTrue(r['frozen_inputs_unchanged'])
            self.assertTrue(r['stock_encode_ids_timed']);self.assertFalse(r['cases']['stock_encode_ids_probe']['raised'])
            self.assertEqual(r['cases']['cpu_fp16_comparison']['embedding_full_numeric']['relative_rms'],0)
    def test_KV_cache_onlyzero_rows_and_exact_parameter_prefix(self):
        r=json.loads((ROOT/'ranks'/'tp8_rank0.json').read_text());p=r['kv_padding_final']
        self.assertEqual(p['cache_entries'],100);self.assertEqual(p['cache_bytes'],262144000)
        self.assertTrue(p['all_prefix_exact']);self.assertEqual(p['tail_nonzero_count'],0)
        self.assertTrue(p['all_cached_padded_are_zero_offset']);self.assertTrue(p['no_cached_padded_holds_autograd_graph'])
        self.assertEqual(p['transform']['matched_calls'],['k_weight','v_weight'])
        self.assertFalse(p['transform']['global_functional_patched'])
    def test_full_encode_ids_timing_matchedrankmax_before_median(self):
        rows=[json.loads((ROOT/'ranks'/('tp8_rank%d.json'%rank)).read_text())for rank in range(8)]
        maxima=[max(r['timing']['samples_synchronized_wall_s_including_d2h'][i]for r in rows)for i in range(3)]
        summary=json.loads((ROOT/'summary81.json').read_text());self.assertEqual(maxima,summary['rank_max_complete_encode_s'])
        self.assertEqual(statistics.median(maxima),summary['rank_max_complete_encode_median_s'])
        self.assertGreater(summary['rank_max_complete_encode_median_s'],2)
    def test_precision_not_waived_despite_repeat_stability(self):
        summary=json.loads((ROOT/'summary81.json').read_text());self.assertTrue(summary['all8_ranks_repeat_bits_equal'])
        self.assertTrue(all(v>.07 for v in summary['CPU_selected50_relative_RMS_by_rank']))
        subsets=summary['selected50_token_subsets_rank0'];self.assertEqual(subsets['text_nonvisual']['npu']['shape'],[27,5120])
        self.assertEqual(subsets['visual']['npu']['shape'],[2016,5120]);self.assertGreater(subsets['reference_abs_lt_512']['relative_rms'],.1)
        self.assertFalse(summary['full_encoder_precision_accepted']);self.assertFalse(summary['whole_goal_complete'])
    def test_corrected_driver_and_original_harness_failure_preserved(self):
        r=json.loads((ROOT/'ranks'/'tp8_rank0.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT/'encode_fp16_padded_kv_fixed81.py').read_bytes()).hexdigest(),r['source_sha256'])
        self.assertEqual(json.loads((ROOT/'run_finished.json').read_text())['exit_code'],1)
        self.assertEqual(json.loads((ROOT/'fixed_run_finished.json').read_text())['exit_code'],0)


if __name__=='__main__':unittest.main()
