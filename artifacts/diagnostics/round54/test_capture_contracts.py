#!/usr/bin/env python3
"""Guard against accidentally changing inference math while adding capture sites."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

def fn(path,name,cls=None):
    tree=ast.parse(path.read_text())
    nodes=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)

class CaptureContracts(unittest.TestCase):
    def test_kernel_build_and_core_math_unchanged(self):
        for name in ('_build','_flash_attention_core','_pad_qkv','_install_key_sequence_limit'):
            self.assertEqual(ast.dump(fn(ROOT/'source/fa910a.py',name)),ast.dump(fn(ROOT/'candidate_fa910a.py',name)))

    def test_sdpa_call_arguments_unchanged(self):
        old=fn(ROOT/'source/encoder.py','_scaled_dot_product_attention')
        new=fn(ROOT/'candidate_encoder.py','_scaled_dot_product_attention')
        before=next(n.value for n in old.body if isinstance(n,ast.Return))
        after=next(n.value for n in new.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='F.scaled_dot_product_attention')
        self.assertEqual(ast.dump(before),ast.dump(after))

    def test_text_and_vision_forward_math_unchanged(self):
        for cls in ('MiniMaxH3Qwen3VLTextModel','MiniMaxH3Qwen3VLTextDecoderLayer','MiniMaxH3Qwen3VLVisionModel'):
            self.assertEqual(ast.dump(fn(ROOT/'source/encoder.py','forward',cls)),ast.dump(fn(ROOT/'candidate_encoder.py','forward',cls)))

    def test_hook_callbacks_always_return_none(self):
        hook=fn(ROOT/'h3_910a_audit.py','_output_hook')
        inner=next(n for n in hook.body if isinstance(n,ast.FunctionDef))
        self.assertIsInstance(inner.body[-1],ast.Return)
        self.assertEqual(ast.dump(inner.body[-1].value),ast.dump(ast.Constant(value=None)))

    def test_measurement_request_is_original_49_step(self):
        text=(ROOT/'e2e_capture_probe.py').read_text()
        self.assertIn('"num_inference_steps": "49"',text)
        self.assertIn('MiniMax-H3-native-fp16-audit53',text)
        self.assertIn('"VLLM_OMNI_910A_AUDIT_DIR": "/tmp/h3_round54_capture01"',text)

if __name__=='__main__':unittest.main(verbosity=2)
