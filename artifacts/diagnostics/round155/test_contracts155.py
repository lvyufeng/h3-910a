#!/usr/bin/env python3
"""Contracts for the R155 additive MLP decomposition."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):
            ast.parse(p.read_text(),filename=str(p))
    def test_probe_keeps_the_native_path(self):
        s=(ROOT/'span_probe155.py').read_text()
        self.assertNotIn('mlp.forward=',s)
        self.assertIn('register_forward_pre_hook',s)
        self.assertIn("if _STATE['installed']:",s)
    def test_single_sync_outside_functions(self):
        tree=ast.parse((ROOT/'span_probe155.py').read_text())
        syncs=[n.lineno for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
        self.assertEqual(len(syncs),1)
    def test_decomposition_actually_closes(self):
        b=json.loads((ROOT/'span155.json').read_text())
        self.assertLess(abs(b['closure_error_ms']),1.0,'the parts must add up to the total')
        self.assertAlmostEqual(b['per_call_sum_median_ms'],b['median_ms']['total'],delta=0.5)
        self.assertTrue(b['additive_by_construction'])
    def test_activation_region_and_the_ten_x_gap(self):
        b=json.loads((ROOT/'span155.json').read_text())
        self.assertGreater(b['median_ms']['activation'],30.0)
        self.assertLess(b['silu_inside_the_activation_ms'],2.0,'the SiLU alone must be small')
        self.assertIn('roughly a tenth',b['contradiction_with_microbench'])
    def test_no_candidate_claimed(self):
        b=json.loads((ROOT/'span155.json').read_text())
        self.assertIn('no candidate',b['not_claimed'])
        self.assertIn('failed',b['probe_attempts']['span155a'])

if __name__=='__main__':unittest.main()
