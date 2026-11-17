"""Validate actual conditioning isolation and unchanged native encoder recovery."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class ActualConditionIsolation(unittest.TestCase):
    def test_all8_full50_layer_cases_match_fixed_source_and_repeat(self):
        summary=json.loads((ROOT/'isolation_summary82.json').read_text())
        self.assertEqual(len(summary['receipts']),8)
        for item in summary['receipts']:
            p=ROOT/'ranks'/('tp8_rank%d.json'%item['rank']);r=json.loads(p.read_text())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
            self.assertTrue(r['completed']);self.assertTrue(r['native_matches_round81_sha256'])
            self.assertTrue(r['native_after_isolation']['raw_bits_equal']);self.assertTrue(r['frozen_inputs_unchanged'])
            self.assertEqual(hashlib.sha256((ROOT/'condition_isolation82.py').read_bytes()).hexdigest(),r['source_sha256'])
            self.assertEqual(len(r['conditioning_isolation']),3)
            for c in r['conditioning_isolation'].values():
                self.assertEqual(len(c['layers']),50);self.assertTrue(c['repeat_vs_first']['raw_bits_equal'])
                self.assertTrue(c['condition_inputs_unchanged']);self.assertTrue(c['not_full_encoder_performance'])
    def test_exact_CPU_fp16_conditions_transferred_bit_identically(self):
        r=json.loads((ROOT/'ranks'/'tp8_rank0.json').read_text());p=r['CPU_condition_transfer_proof']
        self.assertTrue(p['inputs_embeds']['raw_bits_equal'])
        self.assertTrue(all(item['raw_bits_equal']for item in p['deepstack']))
        self.assertTrue(p['position_ids_equal_to_native']);self.assertTrue(p['visual_mask_equal_to_native'])
        self.assertEqual(p['source_sha256'],'6ab37f7202731843aebce9f3f8453232c72d45b2239caba679bcb2611e626e70')
    def test_factor_results_are_measured_not_an_additive_error_or_pass(self):
        s=json.loads((ROOT/'isolation_summary82.json').read_text());f=s['conditioning_factors']
        self.assertAlmostEqual(s['native_selected50_relative_RMS'],.07420123401951458)
        self.assertAlmostEqual(f['CPU_initial_only']['relative_RMS'],.04325680788895705)
        self.assertAlmostEqual(f['CPU_DeepStack_only']['relative_RMS'],.07194147658781876)
        self.assertAlmostEqual(f['CPU_initial_and_DeepStack']['relative_RMS'],.024756999879344863)
        self.assertFalse(s['full_encoder_precision_accepted']);self.assertFalse(s['whole_goal_complete'])
    def test_only_restored_native_fullpath_has_complete_encode_timings(self):
        s=json.loads((ROOT/'isolation_summary82.json').read_text())
        self.assertTrue(s['native_stock_encode_ids_timed']);self.assertGreater(s['native_full_encode_rank_max_median_s'],2)
        for c in s['conditioning_factors'].values():self.assertTrue(c['not_full_encoder_performance'])


if __name__=='__main__':unittest.main()
