#!/usr/bin/env python3
"""Contracts for R145 hook cost and R146 true-shape GEMM (measurement only)."""
import ast,json,unittest
from pathlib import Path
R145=Path(__file__).resolve().parent.parent/'round145'
R146=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in list(R145.glob('*.py'))+list(R146.glob('*.py')):
            ast.parse(p.read_text(),filename=str(p))
    def test_hook_probe_has_no_sync(self):
        s=(R145/'hook_probe145.py').read_text()
        self.assertNotIn('.synchronize(',s)
        self.assertIn('perf_counter',s)
    def test_hook_verdict(self):
        h=json.loads((R145/'hook145.json').read_text())
        self.assertTrue(h['measurement_only'])
        self.assertEqual(h['ranks_completed'],8)
        self.assertLess(h['hook_share_of_block_percent'],1.0)
        self.assertLess(h['per_call_ms']['prefetch_layer'],2.0)
        self.assertIn('not the 3x source',h['verdict'])
    def test_true_shapes_used_not_rescaled(self):
        b=json.loads((R146/'roofline146.json').read_text())
        self.assertTrue(b['measurement_only'])
        self.assertTrue(b['no_model_weights_loaded'])
        self.assertTrue(b['all_ranks_completed'])
        dims={r['case']:(r['M'],r['K'],r['N']) for r in b['rows']}
        self.assertEqual(dims['qkv_proj_true'],(10448,5376,10752))
        self.assertEqual(dims['mlp_fc1_true'],(10448,5376,14336))
        self.assertEqual(dims['attn_out_proj_true'],(10448,3584,5376))
        self.assertEqual(dims['mlp_fc2_true_fp32_input'],(10448,7168,5376))
        self.assertIn('rescaling was invalid',b['note'])
    def test_guard_caveat_recorded(self):
        b=json.loads((R146/'roofline146.json').read_text())
        self.assertIn('FP32 range guard',b['caveat'])
        guarded={r['case']:r['in_model_includes_fp32_range_guard'] for r in b['rows']}
        self.assertTrue(guarded['attn_out_proj_true'] and guarded['mlp_fc2_true_fp32_input'])
        self.assertFalse(guarded['qkv_proj_true'] or guarded['mlp_fc1_true'])
    def test_ruled_out_and_open(self):
        b=json.loads((R146/'roofline146.json').read_text())
        self.assertIn('copy_stall',b['ruled_out'])
        self.assertIn('hook_host_overhead',b['ruled_out'])
        self.assertIn('qkv',b['open'])

if __name__=='__main__':unittest.main()
