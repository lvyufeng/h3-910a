#!/usr/bin/env python3
"""Contracts for the R163 fast-audit equivalence verification."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_verification_only(self):
        b=json.loads((ROOT/'fastaudit163.json').read_text())
        self.assertTrue(b['verification_only'])
        self.assertTrue(b['native_verdict_authoritative_this_run'])
        self.assertTrue(b['no_check_removed'])
        self.assertTrue(b['nothing_adopted'])
    def test_equivalence_is_exact(self):
        b=json.loads((ROOT/'fastaudit163.json').read_text())
        self.assertTrue(b['equivalence']['equivalent_on_every_call'])
        self.assertEqual(b['equivalence']['mismatch_count'],0)
        self.assertEqual(b['calls_compared'],465)
    def test_speedup_is_measured(self):
        b=json.loads((ROOT/'fastaudit163.json').read_text())
        c=b['cost']
        self.assertGreater(c['speedup'],2.0)
        self.assertAlmostEqual(c['speedup'],c['native_median_ms_per_call']/c['fast_median_ms_per_call'],places=1)
    def test_equivalence_argument_recorded(self):
        b=json.loads((ROOT/'fastaudit163.json').read_text())
        self.assertIn('65504',b['why_the_fast_check_is_equivalent'])
        self.assertIn('npu_format_cast',b['why_the_native_check_is_expensive'])
    def test_cross_check_and_residual(self):
        b=json.loads((ROOT/'fastaudit163.json').read_text())
        self.assertIn('2.87',b['cross_check_with_r162'])
        self.assertIn('fusing',b['residual_cost'])

if __name__=='__main__':unittest.main()
