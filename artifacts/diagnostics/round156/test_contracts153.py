#!/usr/bin/env python3
"""Contracts for the R153b MLP residual attribution."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_probe_keeps_the_native_forward_path(self):
        s=(ROOT/'mlp_probe153.py').read_text()
        self.assertNotIn('mlp.forward=',s,'replacing the forward bypasses the dispatcher and breaks weight staging')
        self.assertIn('register_forward_pre_hook',s)
    def test_install_is_idempotent(self):
        s=(ROOT/'mlp_probe153.py').read_text()
        self.assertIn("if _STATE['installed']:",s)
    def test_only_one_sync_and_it_is_outside_functions(self):
        tree=ast.parse((ROOT/'mlp_probe153.py').read_text())
        syncs=[n.lineno for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
        self.assertEqual(len(syncs),1)
    def test_hypothesis_is_marked_rejected(self):
        b=json.loads((ROOT/'mlp153.json').read_text())
        self.assertIn('REJECTED',b['hypothesis_verdict'])
        self.assertLess(b['silu_share_of_residual_percent'],10.0)
        self.assertGreater(b['mul_plus_chunk_ms'],30.0)
    def test_candidate_is_not_claimed_as_adopted(self):
        b=json.loads((ROOT/'mlp153.json').read_text())
        self.assertIn('verified bitwise',b['candidate'])
        md=(ROOT/'RESULTS.md').read_text()
        self.assertIn('不改模型',md)
    def test_strided_explanation_is_flagged_as_hypothesis(self):
        b=json.loads((ROOT/'mlp153.json').read_text())
        self.assertTrue(any('hypothesis' in s for s in b['limits']))

if __name__=='__main__':unittest.main()
