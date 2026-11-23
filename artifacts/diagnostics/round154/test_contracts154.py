#!/usr/bin/env python3
"""Contracts for the R154 strided-view microbench."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_strided_view_really_was_strided(self):
        b=json.loads((ROOT/'micro154.json').read_text())
        self.assertEqual(b['row_stride'],2*b['logical_width'],'chunk halves must have row stride 2N')
    def test_theory_is_rejected_on_net_gain(self):
        b=json.loads((ROOT/'micro154.json').read_text())
        self.assertEqual(b['verdict']['strided_theory'],'REJECTED')
        m=b['ms_median']
        self.assertLess(m['contiguous_silu_then_mul'],m['strided_silu_then_mul'],'operators alone must be faster')
        self.assertAlmostEqual(m['contiguous_plus_copies'],m['strided_silu_then_mul'],delta=0.15,
                               msg='copies must eat the gain')
    def test_bit_identity_recorded(self):
        b=json.loads((ROOT/'micro154.json').read_text())
        self.assertTrue(b['bit_identity']['torch_equal'])
        self.assertEqual(b['bit_identity']['differing_elements'],0)
    def test_in_model_factor_of_ten_is_flagged(self):
        b=json.loads((ROOT/'micro154.json').read_text())
        self.assertIn('factor of about 10',b['verdict']['bigger_finding'])
        self.assertIn('additive decomposition',b['verdict']['attribution_error'])
    def test_no_model_change_claimed(self):
        b=json.loads((ROOT/'micro154.json').read_text())
        self.assertTrue(b['measurement_only'])
        self.assertTrue(b['no_model_loaded'])

if __name__=='__main__':unittest.main()
