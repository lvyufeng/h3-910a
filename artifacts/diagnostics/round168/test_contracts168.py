#!/usr/bin/env python3
"""Contracts for the R168 rejection of the cheap audit candidate."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_rejected_on_measurement(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        self.assertEqual(b['decision'],'REJECT')
        self.assertGreater(b['delta_s'],0.0)
    def test_checks_still_ran(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        c=b['checks_still_running']
        self.assertTrue(c['all_155_checks_still_run'])
        self.assertTrue(c['model_arithmetic_unchanged'])
        self.assertEqual(c['calls_checked'],465)
        self.assertEqual(c['flagged'],0)
    def test_contradiction_is_stated(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        self.assertLess(b['check_cost']['device_s_per_step'],6.656)
        self.assertIn('did not translate',b['the_contradiction'])
    def test_the_lesson_is_recorded(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        self.assertIn('serialisation point',b['what_this_teaches'])
        self.assertIn('not simply the audits arithmetic time',b['what_this_teaches'])
    def test_prior_estimate_withdrawn(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        self.assertIn('0.65',b['consequence_for_the_status_document'])
        self.assertIn('withdrawn',b['consequence_for_the_status_document'])
    def test_no_user_decision_needed_anymore(self):
        b=json.loads((ROOT/'audit_candidate168.json').read_text())
        self.assertIn('none needed',b['decision_for_the_user'])

if __name__=='__main__':unittest.main()
