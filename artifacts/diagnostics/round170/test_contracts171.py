#!/usr/bin/env python3
"""Contracts for the R171 maximal-combination bound."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_bound_is_above_target(self):
        b=json.loads((ROOT/'maxbound171.json').read_text())
        self.assertGreater(b['maximal_result_s'],10.0)
        self.assertIn('NOT reachable',b['verdict'])
    def test_every_region_is_accounted(self):
        b=json.loads((ROOT/'maxbound171.json').read_text())
        self.assertEqual(len(b['regions']),8)
        total_now=sum(v.get('now_s',0) for v in b['regions'].values())
        self.assertGreater(total_now,17.0)
    def test_caveats_are_recorded(self):
        b=json.loads((ROOT/'maxbound171.json').read_text())
        self.assertIn('NO mechanism',b['regions']['usp_pre_post']['caveat'])
        self.assertIn('NO mechanism',b['regions']['attention_residual']['caveat'])
        self.assertIn('precision decision',b['regions']['four_linears']['caveat'])
    def test_audit_is_irreducible_with_evidence(self):
        b=json.loads((ROOT/'maxbound171.json').read_text())
        w=b['regions']['safety_audit']['why']
        self.assertIn('R168',w)
        self.assertIn('R170',w)
        self.assertEqual(b['regions']['safety_audit']['saving_s'],0.0)
    def test_what_would_be_needed_is_stated(self):
        b=json.loads((ROOT/'maxbound171.json').read_text())
        self.assertEqual(len(b['what_would_be_needed']),3)

if __name__=='__main__':unittest.main()
