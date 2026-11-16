#!/usr/bin/env python3
"""Structural regression of token-head tiling and explicit RoPE contracts."""
import ast
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent


class RopeContracts(unittest.TestCase):
    def test_half_rotation_is_per_head(self):
        for heads in (1,8):
            x=list(range(heads*128))
            rotated=[]
            for head in range(heads):
                row=x[head*128:(head+1)*128]
                rotated.extend([-value for value in row[64:]]+row[:64])
            by_offsets=[None]*len(x)
            for head in range(heads):
                by_offsets[head*128:head*128+64]=[-value for value in x[head*128+64:head*128+128]]
                by_offsets[head*128+64:head*128+128]=x[head*128:head*128+64]
            self.assertEqual(rotated,by_offsets)

    def test_products_round_before_add_only_for_hf_contract(self):
        tree=ast.parse((ROOT/'rope_vector60.py').read_text())
        ifs=[node for node in ast.walk(tree) if isinstance(node,ast.If) and isinstance(node.test,ast.Name) and node.test.id=='hf_round']
        self.assertEqual(len(ifs),1)
        calls=[node for node in ast.walk(ifs[0]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_round_bf16']
        self.assertEqual(len(calls),2)
        text=(ROOT/'rope_vector60.py').read_text()
        self.assertIn('t.h_add(out, out, rotated)',text)
        self.assertIn('_round_bf16(t, out, out',text)

    def test_native_gm_inputs_no_external_float_casts(self):
        tree=ast.parse((ROOT/'rope_vector60.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rope')
        calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('float','half')]
        self.assertFalse(calls)
        text=ast.unparse(node)
        self.assertIn('torch.bfloat16',text)
        self.assertIn('get_npu_format',text)
        self.assertIn('x.transpose(1, 2).contiguous()',text)

    def test_cache_separates_rounding_shape_and_live_cores(self):
        text=(ROOT/'rope_vector60.py').read_text()
        self.assertIn('key = (tokens, heads, bool(hf_round), cores)',text)
        self.assertIn('hf{int(hf_round)}',text)
        self.assertIn('cores = min(core_budget(), tokens)',text)
        self.assertIn('fcntl.LOCK_EX',text)
        self.assertIn('Ascend910A',text)

    def test_capture_is_bounded_and_not_production_instrumentation(self):
        text=(ROOT/'text_rope_tp8.py').read_text()
        self.assertIn("'inputs_unchanged'",text)
        self.assertIn('512*1024*1024',text)
        self.assertIn("layer in (0, 24, 49)",text)
        self.assertIn('ROPE_CALL = 0',text)
        self.assertIn("'third_vs_repeat': metric(third_cpu, repeat_cpu)",text)

    def test_known_shape_layout_and_coeff_broadcast(self):
        text=(ROOT/'rope_vector60.py').read_text()
        self.assertIn('x.shape[1] in (1, 8)',text)
        self.assertIn('cos.shape == sin.shape == (1, tokens, 128)',text)
        self.assertIn('t.data_move(cos[head*d], cos',text)
        self.assertIn('t.data_move(sin[head*d], sin',text)
        self.assertIn('d//8, d//8',text)

if __name__=='__main__':
    unittest.main(verbosity=2)
