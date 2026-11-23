#!/usr/bin/env python3
"""Contracts for the R142 standalone microbenchmark (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_no_model_and_no_math_change(self):
        s=(ROOT/'microbench142.py').read_text()
        for banned in ('MiniMaxH3DiTModel','load_weights','omni.generate(','safe_open',
                       'set_forward_context','install_' ):
            self.assertNotIn(banned,s)
    def test_shapes_are_the_measured_ones(self):
        s=(ROOT/'microbench142.py').read_text()
        for token in ('41792','USP = 4','D, HEADS, HD, FFN, TP = 5376, 56, 128, 14336, 2'):
            self.assertIn(token,s)
    def test_guard_unresolved_is_recorded_not_guessed(self):
        b=json.loads((ROOT/'roofline142.json').read_text())
        guard=b['guard_split_unresolved']
        self.assertIn('ReduceMax',guard)
        self.assertIn('no estimate',guard)
    def test_conclusion_and_numbers(self):
        b=json.loads((ROOT/'roofline142.json').read_text())
        self.assertTrue(b['measurement_only'])
        self.assertTrue(b['no_model_weights_loaded'])
        self.assertTrue(b['all_ranks_completed'])
        self.assertGreater(b['gemm_attainable_saving_s'],3.0)
        self.assertLess(b['practical_gemm_ceiling_TFLOPS_per_card'],120)
        self.assertEqual(len(b['in_model_vs_standalone']),4)
        worst=b['in_model_vs_standalone'][0]
        self.assertEqual(worst['case'],'qkv_proj_resident')
        self.assertGreater(worst['in_model_vs_standalone_x'],5.0)
    def test_fa_standalone_honestly_skipped(self):
        b=json.loads((ROOT/'roofline142.json').read_text())
        self.assertIn('cannot reveal a faster variant',b['fa_standalone'])

if __name__=='__main__':unittest.main()
