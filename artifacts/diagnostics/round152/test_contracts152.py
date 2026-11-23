#!/usr/bin/env python3
"""Contracts for the R152 SiLU probe (measurement only, result inconclusive)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_probe_only_times(self):
        s=(ROOT/'silu_probe152.py').read_text()
        self.assertIn('torch.npu.Event',s)
        tree=ast.parse(s)
        syncs=[n.lineno for n in ast.walk(tree)
               if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='synchronize']
        self.assertEqual(len(syncs),1,'one sync, after the step')
    def test_result_is_labelled_inconclusive(self):
        b=json.loads((ROOT/'silu152.json').read_text())
        self.assertEqual(b['hypothesis_verdict'],'PARTIALLY REJECTED')
        self.assertIn('sum',b['duplication_evidence'].lower()+json.dumps(b))
        md=(ROOT/'RESULTS.md').read_text()
        self.assertIn('INCONCLUSIVE',md)
    def test_defects_are_named(self):
        b=json.loads((ROOT/'silu152.json').read_text())
        self.assertEqual(b['hook_duplication_factor'],3)
        self.assertGreater(b['silu']['max_ms'],100,'the outlier that pollutes the sum must be recorded')
        self.assertGreater(len(b['silu']['shapes_seen']),2,'mixed shapes are why the median cannot represent the MLP call')
    def test_mlp_median_matches_r149(self):
        b=json.loads((ROOT/'silu152.json').read_text())
        self.assertAlmostEqual(b['mlp']['median_ms'],88.178,delta=1.0)
    def test_s6_not_claimed_as_verified(self):
        md=(ROOT/'RESULTS.md').read_text()
        self.assertIn('未验证',md)

if __name__=='__main__':unittest.main()
