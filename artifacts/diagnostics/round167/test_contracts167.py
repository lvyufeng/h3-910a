#!/usr/bin/env python3
"""Contracts for the R167 in-model verification of the bit-pattern check."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_exact_equivalence_in_the_model(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        self.assertTrue(b['equivalence']['exactly_count_equivalent_on_every_call'])
        self.assertEqual(b['equivalence']['mismatch_count'],0)
        self.assertEqual(b['calls_compared'],465)
    def test_combined_with_the_adversarial_suite(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        self.assertIn('R166',b['equivalence']['combined_with_r166'])
        self.assertIn('37',b['equivalence']['combined_with_r166'])
    def test_median_paradox_is_explained(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        c=b['cost']
        self.assertGreater(c['bit_median_ms'],c['native_median_ms'])
        self.assertLess(c['bit_device_s_per_step'],c['native_device_s_per_step'])
        self.assertIn('heavier tail',b['the_median_paradox'])
    def test_expectation_corrected_downwards(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        e=b['corrected_expectation_for_step_time']
        self.assertIn('0.6 to 0.7',e)
        self.assertIn('not the 1.5 to 2 s',e)
    def test_remaining_cost_stated(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        self.assertIn('4.808',b['remaining_audit_cost'])
        self.assertIn('fused',b['remaining_audit_cost'])
    def test_verification_only(self):
        b=json.loads((ROOT/'bit167.json').read_text())
        self.assertTrue(b['verification_only'])
        self.assertTrue(b['nothing_adopted'])

if __name__=='__main__':unittest.main()
