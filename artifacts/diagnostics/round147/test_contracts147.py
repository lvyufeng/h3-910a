#!/usr/bin/env python3
"""Contracts for the R147 device-event attribution (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_no_sync_in_forward(self):
        # AST check so prose in a docstring cannot satisfy or break the contract.
        tree=ast.parse((ROOT/'event_probe147.py').read_text())
        syncs=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) \
                    and node.func.attr=='synchronize':
                syncs.append(getattr(node,'lineno',-1))
        self.assertEqual(len(syncs),1,'exactly one sync is allowed, after the measured step')
        enclosing=[n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,))]
        self.assertGreater(syncs[0],max(enclosing),'the only sync must sit outside every function')
    def test_targets_cover_the_linears(self):
        s=(ROOT/'event_probe147.py').read_text()
        for t in ("'qkv','attn.qkv_proj'","'out_proj','attn.out_proj'","'fc1','mlp.fc1'","'fc2','mlp.fc2'"):
            self.assertIn(t,s)
    def test_numbers(self):
        b=json.loads((ROOT/'attribution147.json').read_text())
        self.assertTrue(b['measurement_only'] and b['no_sync_inside_forward'])
        self.assertTrue(b['all_ranks_completed'])
        rows={r['label']:r for r in b['linears']}
        self.assertLess(rows['qkv']['ratio'],2.0)
        self.assertLess(rows['fc1']['ratio'],2.0)
        self.assertGreater(rows['out_proj']['ratio'],5.0)
        self.assertTrue(rows['out_proj']['includes_fp32_range_guard'])
        self.assertGreater(b['share_of_gap_in_guarded_pair_percent'],70)
        self.assertIn('sync',b['correction'].lower())
    def test_guard_claim_is_conditional_not_assumed(self):
        b=json.loads((ROOT/'attribution147.json').read_text())
        text=b['bit_identical_opportunity']
        self.assertIn('must be proven, not assumed',text)
        self.assertIn('32752',text)

if __name__=='__main__':unittest.main()
