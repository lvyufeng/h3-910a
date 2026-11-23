#!/usr/bin/env python3
"""Contracts for the R151 corrected feasibility arithmetic."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_components_sum_to_the_step(self):
        b=json.loads((ROOT/'ceiling_table151.json').read_text())
        total=sum(b['components_s'].values())
        self.assertAlmostEqual(total,b['step_s'],delta=0.35)
    def test_fa_flops_and_rate(self):
        b=json.loads((ROOT/'ceiling_table151.json').read_text())
        self.assertAlmostEqual(b['fa_flops_per_rank_per_step_TF'],313.0,delta=1.0)
        self.assertAlmostEqual(b['fa_achieved_TFLOPS_per_card'],313.0/b['components_s']['fa'],delta=1.0)
    def test_correction_is_explicit(self):
        b=json.loads((ROOT/'ceiling_table151.json').read_text())
        self.assertIn('logically',b['correction'])
        self.assertIn('10.42',b['correction'])
    def test_single_levers_are_insufficient(self):
        S=json.loads((ROOT/'ceiling_table151.json').read_text())['scenarios_s']
        for k in ('S1 FA to zero, rest unchanged','S2 FA 2x, rest unchanged','S4 FA 2x + linears standalone'):
            self.assertGreater(S[k],10.0,f'{k} must not already reach 10 s')
    def test_combination_can_reach_ten(self):
        S=json.loads((ROOT/'ceiling_table151.json').read_text())['scenarios_s']
        combined=next(v for k,v in S.items() if k.startswith('S6'))
        self.assertLess(combined,10.0,'the combined scenario must show 10 s is not excluded')
    def test_fa_requirement_is_below_best_measured(self):
        W=json.loads((ROOT/'ceiling_table151.json').read_text())['what_10s_requires']
        self.assertLess(W['if_linears_improve_and_five_overheads_halve']['vs_best_measured'],1.0,
                        'the reachable path must need less than the best measured rate')
        self.assertGreater(W['if_only_linears_improve']['vs_best_measured'],1.5,
                           'the single-lever path must be the implausible one')

if __name__=='__main__':unittest.main()
