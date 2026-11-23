#!/usr/bin/env python3
"""Contracts for the R149 complete step budget (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_no_sync_in_forward(self):
        s=(ROOT/'budget_probe149.py').read_text()
        tree=ast.parse(s)
        syncs=[n.lineno for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
        self.assertEqual(len(syncs),1,'exactly one sync, after the measured step')
        fns=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
        self.assertGreater(syncs[0],max(fns))
    def test_wrapper_factor_recorded_not_ignored(self):
        b=json.loads((ROOT/'budget149.json').read_text())
        self.assertGreater(b['fa_wrapper_install_factor'],1.0)
        self.assertIn('recorded 3 times',b['fa_wrapper_note'])
        # the corrected FA must be smaller than the raw recorded sum
        self.assertLess(b['budget_ms']['fa'],24600)
    def test_budget_internal_consistency(self):
        b=json.loads((ROOT/'budget149.json').read_text())
        B=b['budget_ms']
        self.assertAlmostEqual(B['linears_total'],
                               sum(B['linears'].values()),places=1)
        self.assertLess(B['block_residual'],B['block_total'])
        self.assertGreater(B['step_front_tail_residual'],0)
    def test_ten_second_verdict_is_arithmetic(self):
        b=json.loads((ROOT/'budget149.json').read_text())
        A=b['ten_second_arithmetic']
        self.assertAlmostEqual(A['needed_reduction_s'],round(b['step_s']-10.0,3),places=3)
        self.assertGreater(A['fa_elimination_still_leaves_s'],9.5)
        self.assertIn('cannot be reached',A['verdict'])
        self.assertGreater(b['optimistic_projection']['fa_2x_s'],10.0)

if __name__=='__main__':unittest.main()
