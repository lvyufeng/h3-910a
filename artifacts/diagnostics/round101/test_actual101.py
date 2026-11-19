"""Actual BC128 phantom-padding and constant CPU-Half smoke proof."""
from pathlib import Path
import json
import hashlib
import unittest
ROOT=Path(__file__).resolve().parent


class Actual101Smoke(unittest.TestCase):
    def test_failed_oneelement_slice_is_preserved_successful_corrected_smoke_complete(self):
        old=json.loads((ROOT/'results/smoke01/bc128_smoke101.json').read_text());new=json.loads((ROOT/'results/smoke02/bc128_smoke101.json').read_text())
        self.assertFalse(old['completed']);self.assertTrue(new['completed'])
        self.assertEqual(new['meta']['block_dim'],4);self.assertEqual(new['meta']['GM_argument_count'],5)
        self.assertEqual(new['meta']['source_sha256'],hashlib.sha256((ROOT/'flash_vision_mask_fixed101.py').read_bytes()).hexdigest())
    def test_poison_padding_is_bitwise_invariant_in_both_actual_controls(self):
        r=json.loads((ROOT/'results/smoke02/bc128_smoke101.json').read_text());self.assertTrue(r['completed'])
        for case in r['cases'].values():
            self.assertTrue(case['poison_vs_zero_pad']['raw_bits_equal'])
            for row in case['padding'].values():
                self.assertTrue(row['all_repeats_and_inputs_unchanged'])
                self.assertTrue(row['vs_CPU_Half_MATH']['finite']);self.assertEqual(len(row['clock_s']),5)
        constant=r['cases']['constant_uniform']
        self.assertTrue(constant['padding']['zero']['vs_CPU_Half_MATH']['raw_bits_equal'])
        self.assertTrue(constant['padding']['poison']['vs_CPU_Half_MATH']['raw_bits_equal'])
        self.assertFalse(r['full_encoder_precision_accepted']or r['whole_goal_complete'])
        self.assertIsNone(r['numerical_tolerance'])


if __name__=='__main__':unittest.main()
