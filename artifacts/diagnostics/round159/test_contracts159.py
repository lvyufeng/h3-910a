#!/usr/bin/env python3
"""Contracts for the R159 read-only FA plan analysis."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_read_only(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        self.assertTrue(b['read_only'])
        self.assertTrue(b['no_model_loaded'])
        self.assertTrue(b['no_code_changed'])
    def test_wrong_suspicion_is_retracted(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        r=b['retraction']
        self.assertIn('retracted before being proposed',r['status'])
        self.assertIn('43776',r['why_it_is_wrong'])
        self.assertIn('128',r['why_it_is_wrong'])
    def test_plan_is_evidence_based(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        self.assertIn('Br=128',b['confirmed_launch_plan'])
        self.assertIn('Bc=304',b['confirmed_launch_plan'])
        self.assertIn('historical log',b['plan_source'])
    def test_padding_arithmetic(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        p=b['padding_arithmetic']
        self.assertEqual(p['kernel_N'],43776)
        self.assertGreater(p['wasted_flops_percent'],9.0)
        self.assertAlmostEqual(p['tiles_now'],49248)
        self.assertAlmostEqual(p['tiles_without_padding_at_the_same_tile_shape'],45126)
    def test_candidate_saving_matches_arithmetic(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        c=b['candidate_identified']
        self.assertAlmostEqual(c['expected_saving_s_per_step'],round(0.0972*7.607,2),places=2)
        self.assertGreater(c['expected_saving_s_per_step'],0.5)
    def test_limits_are_stated(self):
        b=json.loads((ROOT/'fa_plan159.json').read_text())
        joined=' '.join(b['what_this_does_not_do'])
        self.assertIn('69 TFLOPs',joined)
        self.assertIn('new kernel work',b['conclusion'])

if __name__=='__main__':unittest.main()
