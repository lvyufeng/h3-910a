#!/usr/bin/env python3
"""Contracts for the R162 audit-cost diagnostic."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_marked_diagnostic_only(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        self.assertTrue(b['diagnostic_only']); self.assertTrue(b['never_a_candidate'])
        self.assertTrue(b['model_arithmetic_unchanged'])
    def test_region_collapses_to_standalone(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        c=b['comparison_with_strict_audit']
        self.assertLess(c['audit_disabled_activation_ms'],5.0,'the region must collapse to about the standalone 3.24 ms')
        self.assertGreater(c['activation_delta_ms'],-32.0)
        self.assertGreater(c['step_delta_s'],-3.5)
    def test_fc1_fc2_unchanged(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        c=b['comparison_with_strict_audit']
        self.assertAlmostEqual(c['fc1_strict'],c['fc1_audit_disabled'],delta=1.0)
        self.assertAlmostEqual(c['fc2_strict'],c['fc2_audit_disabled'],delta=1.0)
    def test_attribution_is_resolved(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        self.assertIn('RESOLVED',b['attribution'])
        self.assertIn('E1',b['why_it_was_missed'])
    def test_checks_are_not_removed_by_the_candidate(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        self.assertIn('does not remove the checks',b['honest_limit'])
        self.assertIn('155',b['candidate_shape'])
    def test_feasibility_update_reaches_below_ten(self):
        b=json.loads((ROOT/'audit162.json').read_text())
        self.assertLess(b['feasibility_update']['fa_2x_plus_linears_at_standalone_s'],10.0)

if __name__=='__main__':unittest.main()
