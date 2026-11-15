#!/usr/bin/env python3
"""Small unmasked vision-attention regressions; no model assets or dtype changes."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def load_fn():
    tree = ast.parse((ROOT / "candidate_encoder.py").read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "_vision_sdpa_bf16_compatible")
    scope = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "tested-vision-attention", "exec"), scope)
    return scope[node.name]


class VisionAttentionTests(unittest.TestCase):
    def test_cpu_path_unchanged(self):
        fn = load_fn()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            q, k, v = [torch.randn(1, 4, 32, 72, dtype=dtype) * .1 for _ in range(3)]
            ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
            self.assertTrue(torch.equal(fn(q, k, v), ref))

    def test_npu_bf16_head72_and_irregular_lengths(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        fn = load_fn()
        torch.manual_seed(50)
        for qlen, klen, d in ((64, 64, 72), (80, 93, 72), (16, 32, 128)):
            q = torch.randn(1, 4, qlen, d, dtype=torch.bfloat16) * .1
            k = torch.randn(1, 4, klen, d, dtype=torch.bfloat16) * .1
            v = torch.randn_like(k)
            ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=False)
            out = fn(*(t.to("npu") for t in (q, k, v)))
            torch.npu.synchronize()
            got = out.cpu()
            err = (got.float() - ref.float()).abs()
            rel = err.square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-12)
            self.assertLess(rel.item(), .005)
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(out.shape, q.shape)
            self.assertTrue(torch.isfinite(got).all())

    def test_npu_zero_query_uniform_average(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        q = torch.zeros(1, 2, 4, 72, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 8, 72, dtype=torch.bfloat16)
        v = torch.arange(8, dtype=torch.bfloat16).view(1, 1, 8, 1).expand(1, 2, 8, 72).contiguous()
        out = load_fn()(*(t.to("npu") for t in (q, k, v)))
        self.assertTrue(torch.equal(out.cpu(), torch.full_like(q, 3.5)))
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
