#!/usr/bin/env python3
"""Contracts for the R158 synchronous-prefetch diagnostic."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_control_is_marked_diagnostic_only(self):
        s=(ROOT/'run_sync158.py').read_text()
        self.assertIn('DIAGNOSTIC ONLY',s)
        self.assertIn('never proposed as an optimisation',s)
        self.assertIn('non_blocking=False',s)
    def test_control_wiring(self):
        b=json.loads((ROOT/'sync158.json').read_text())
        self.assertTrue(b['diagnostic_control_only'])
        self.assertTrue(b['never_a_candidate'])
    def test_contention_rejected_with_the_delta(self):
        b=json.loads((ROOT/'sync158.json').read_text())
        self.assertEqual(b['contention_hypothesis'],'REJECTED')
        self.assertLess(abs(b['comparison']['activation_delta_ms']),0.5,
                        'the region must not move if copies are the cause')
        self.assertGreater(b['step_s']-b['comparison']['async_prefetch_step_s'],0.5,
                           'the step must get slower, proving copies stopped overlapping')
    def test_all_four_hypotheses_listed(self):
        b=json.loads((ROOT/'sync158.json').read_text())
        self.assertEqual(len(b['hypotheses_now_rejected']),4)
    def test_pivot_is_recorded(self):
        b=json.loads((ROOT/'sync158.json').read_text())
        self.assertIn('critical path',b['decision'])

if __name__=='__main__':unittest.main()
