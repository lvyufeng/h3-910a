#!/usr/bin/env python3
"""Contracts for the R150 event-level FA decomposition (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_runs_baseline_mode_with_profile_audit(self):
        s=(ROOT/'run_fa150.py').read_text()
        self.assertIn("'--mode','baseline'",s)
        self.assertIn("'--audit-sequence','profile'",s)
        self.assertIn('ActualAttentionProfile126',s)
    def test_numbers_are_consistent(self):
        b=json.loads((ROOT/'fa150.json').read_text())
        B=b['ms_per_step']
        self.assertLess(B['FA_ACL_kernel'],B['FA_core'])
        self.assertLess(B['FA_core'],B['FA_wrapper_inclusive'])
        self.assertAlmostEqual(B['FA_wrapper_overhead'],
                               B['FA_wrapper_inclusive']-B['FA_ACL_kernel'],places=1)
        self.assertAlmostEqual(B['USP_total'],
                               B['USP_pre']+B['USP_post'],places=0)
    def test_wrapper_correction_is_recorded(self):
        b=json.loads((ROOT/'fa150.json').read_text())
        self.assertIn('only about 130 ms',b['corrections']['wrapper_overhead'])
        self.assertIn('precise',b['corrections']['wrapper_overhead'])
    def test_ten_second_still_unreachable(self):
        b=json.loads((ROOT/'fa150.json').read_text())
        A=b['ten_second_arithmetic']
        self.assertGreater(A['fa_kernel_to_zero_leaves_s'],10.0)
        self.assertIn('zero-cost attention',A['verdict'])
        self.assertIn('0.42 s',A['verdict'])
        self.assertIn('unreachable',b['conclusion'])
    def test_free_headroom_is_small(self):
        b=json.loads((ROOT/'fa150.json').read_text())
        N=b['numerically_free_headroom']
        self.assertLess(N['attention_internal_residual']+N['fa_wrapper_overhead'],2500)
        self.assertIn('far short',N['note'])

if __name__=='__main__':unittest.main()
