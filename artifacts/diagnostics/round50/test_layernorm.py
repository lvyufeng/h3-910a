#!/usr/bin/env python3
"""Regressions for the exact BF16-compatible norm class without loading H3."""
import ast
from pathlib import Path
import types
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def load_class():
    tree = ast.parse((ROOT / "candidate_encoder.py").read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "_NpuBF16CompatibleLayerNorm")
    scope = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "tested-norm-class", "exec"), scope)
    return scope[node.name]


def reference64(x, layer):
    dims = tuple(range(x.ndim - len(layer.normalized_shape), x.ndim))
    xd = x.double()
    center = xd - xd.mean(dim=dims, keepdim=True)
    y = center / torch.sqrt(center.square().mean(dim=dims, keepdim=True) + layer.eps)
    if layer.weight is not None:
        y *= layer.weight.detach().cpu().double()
    if layer.bias is not None:
        y += layer.bias.detach().cpu().double()
    return y.to(x.dtype)


class StableNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Norm = load_class()

    def layer(self, shape=1152, dtype=torch.bfloat16, affine=True):
        layer = self.Norm(shape, eps=1e-6, elementwise_affine=affine, dtype=dtype)
        if affine:
            torch.manual_seed(50)
            layer.weight.data.copy_(torch.randn_like(layer.weight) * .05 + 1)
            layer.bias.data.copy_(torch.randn_like(layer.bias) * .01)
        return layer

    def assert_close(self, got, ref):
        got_cpu = got.detach().cpu()
        err = (got_cpu.float() - ref.float()).abs()
        rel = err.square().mean().sqrt() / ref.float().square().mean().sqrt().clamp_min(1e-12)
        self.assertLess(rel.item(), .001)
        self.assertLessEqual(err.max().item(), .0625)
        # This driver's native IsFinite does not accept BF16 on NPU; validate
        # copied output on CPU so the harness cannot poison the device context.
        self.assertTrue(torch.isfinite(got_cpu).all())

    def test_cpu_native_branches_bit_exact(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            layer = self.layer(dtype=dtype)
            x = torch.randn(8, 1152, dtype=dtype)
            self.assertTrue(torch.equal(layer(x), F.layer_norm(x, layer.normalized_shape, layer.weight, layer.bias, layer.eps)))

    def test_parameter_names_and_loading_unchanged(self):
        layer = self.layer()
        self.assertIsInstance(layer, nn.LayerNorm)
        self.assertEqual(set(layer.state_dict()), {"weight", "bias"})
        native = nn.LayerNorm(1152, eps=1e-6, dtype=torch.bfloat16)
        native.load_state_dict(layer.state_dict(), strict=True)
        self.assertTrue(torch.equal(native.weight, layer.weight))

    def test_constant_and_zero_are_exact_bias(self):
        layer = self.layer()
        for constant in (0., 13., 131072.):
            x = torch.full((16, 1152), constant, dtype=torch.bfloat16)
            self.assertTrue(torch.equal(layer._stable_bf16_forward(x), layer.bias.expand_as(x)))

    def test_high_offset_retains_bf16_dynamic_range(self):
        layer = self.layer()
        x = (torch.randn(16, 1152) * 8192 + 131072).to(torch.bfloat16)
        self.assert_close(layer._stable_bf16_forward(x), reference64(x, layer))

    def test_wide_postshuffle_norm(self):
        layer = self.layer(4608)
        x = torch.randn(16, 4608, dtype=torch.bfloat16)
        self.assert_close(layer._stable_bf16_forward(x), reference64(x, layer))

    def test_multidimensional_normalized_shape(self):
        layer = self.layer((2, 576))
        x = torch.randn(4, 2, 576, dtype=torch.bfloat16)
        self.assert_close(layer._stable_bf16_forward(x), reference64(x, layer))

    def test_no_affine(self):
        layer = self.layer(affine=False)
        x = torch.randn(4, 1152, dtype=torch.bfloat16)
        self.assert_close(layer._stable_bf16_forward(x), reference64(x, layer))
        self.assertTrue(torch.equal(layer._stable_bf16_forward(torch.ones_like(x)), torch.zeros_like(x)))

    def test_npu_normal_and_constant_and_wide(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        for width in (1152, 4608):
            layer = self.layer(width)
            x = torch.randn(16, width, dtype=torch.bfloat16)
            expected = reference64(x, layer)
            layer = layer.to("npu")
            out = layer(x.to("npu"))
            torch.npu.synchronize()
            self.assert_close(out, expected)
            self.assertEqual(out.dtype, torch.bfloat16)
            constant = torch.full((16, width), 131072., dtype=torch.bfloat16, device="npu")
            constant_out = layer(constant).detach().cpu()
            constant_ref = layer.bias.detach().cpu().expand_as(constant_out)
            self.assertTrue(torch.equal(constant_out, constant_ref))
            torch.npu.synchronize()

    def test_npu_noncontiguous_and_multidimensional(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        layer = self.layer((2, 576))
        x = torch.randn(4, 576, 2, dtype=torch.bfloat16).transpose(1, 2)
        expected = reference64(x, layer)
        xn = x.to("npu")
        layer = layer.to("npu")
        self.assert_close(layer(xn), expected)
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
