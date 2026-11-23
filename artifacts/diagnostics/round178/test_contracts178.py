#!/usr/bin/env python3
"""Contracts for the R178 proxy measurement."""
import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_covers_all_three_audited_shapes(self):
        b=json.loads((ROOT/'proxy178_receipt.json').read_text())
        names={r['tensor'] for r in b['rows']}
        self.assertEqual(names,{'mlp_fc1','qkv_proj','adaln_linear'})
        self.assertEqual(sum(r['calls_per_step'] for r in b['rows']),155)
    def test_saving_is_measured_not_assumed(self):
        b=json.loads((ROOT/'proxy178_receipt.json').read_text())
        self.assertGreater(b['per_step_total_s'],0.0)
        self.assertAlmostEqual(b['implied_saving_s_per_step'],
            2.87-b['per_step_total_s'],places=2)
    def test_bounds_are_stated_not_proof(self):
        b=json.loads((ROOT/'proxy178_receipt.json').read_text())
        self.assertIn('does not prove',b['bounds_not_proof'])
    def test_measurement_only(self):
        b=json.loads((ROOT/'proxy178_receipt.json').read_text())
        self.assertTrue(b['read_only_measurement'])
        self.assertTrue(b['nothing_adopted'])
        self.assertTrue(b['model_arithmetic_unchanged'])
    def test_op_quirk_recorded(self):
        b=json.loads((ROOT/'proxy178_receipt.json').read_text())
        self.assertIn('aclrtAllocatorGetByStream',b['known_op_quirk'])

if __name__=='__main__':unittest.main()
