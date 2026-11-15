#!/usr/bin/env python3
"""Exact CPU/NPU padding tests for the real patched helper and Downsample3D AST."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class ConvStub(nn.Module):
    def __init__(self, *args, padding_mode="zeros", causal=True, **kwargs):
        super().__init__()
        self.causal = causal
        self.pad_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.spatial_parallel = False
        self.chunk_dim = 3

    def forward(self, x):
        return x


def load():
    tree = ast.parse((ROOT / "candidate_vae_cnn.py").read_text())
    names = {"_pad_base_format", "Downsample3D"}
    nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
    scope = {"F": F, "nn": nn, "SpatialParallelConv3d": ConvStub,
             "get_parallel_state": lambda: {"sp_rank": 0, "sp_size": 1, "sp_process_group": None},
             "exchange_strides": lambda x, *args: x + 3}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "real-pad-helper", "exec"), scope)
    return scope["_pad_base_format"], scope["Downsample3D"]


class PaddingTests(unittest.TestCase):
    def test_cpu_reflection_exact(self):
        fn, _ = load()
        x = torch.randn(2, 3, 2, 7, 9)
        pad = (0, 1, 0, 1, 0, 0)
        self.assertTrue(torch.equal(fn(x, pad, "reflect"), F.pad(x, pad, mode="reflect")))

    def test_cpu_other_modes_unchanged(self):
        fn, _ = load()
        x = torch.randn(1, 3, 2, 7, 9)
        for mode in ("constant", "replicate"):
            pad = (1, 1, 0, 1, 0, 0)
            self.assertTrue(torch.equal(fn(x, pad, mode), F.pad(x, pad, mode=mode)))

    def test_downsample_helper_is_wired_to_real_forward(self):
        _, cls = load()
        x = torch.randn(1, 3, 2, 7, 9)
        layer = cls(3, 3, padding_mode="reflect")
        self.assertTrue(torch.equal(layer(x), F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")))

    def test_no_space_downsample_preserves_input(self):
        _, cls = load()
        x = torch.randn(1, 3, 2, 7, 9)
        layer = cls(3, 3, space_stride=1, padding_mode="reflect")
        self.assertTrue(torch.equal(layer(x), x))

    def test_spatial_exchange_branch_unchanged(self):
        _, cls = load()
        x = torch.randn(1, 3, 2, 7, 9)
        layer = cls(3, 3, padding_mode="reflect")
        layer.conv.spatial_parallel = True
        self.assertTrue(torch.equal(layer(x), x + 3))

    def test_npu_packed_fp32_and_fp16_are_exact(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        fn, cls = load()
        for dtype in (torch.float32, torch.float16):
            for shape in ((1, 3, 2, 7, 9), (1, 16, 1, 8, 10), (2, 17, 3, 8, 10)):
                x = torch.randn(shape).to(dtype)
                xn = torch_npu.npu_format_cast(x.to("npu"), 32)
                original = xn.cpu()
                layer = cls(shape[1], shape[1], padding_mode="reflect")
                out = layer(xn)
                torch.npu.synchronize()
                expected = F.pad(x.float(), (0, 1, 0, 1, 0, 0), mode="reflect").to(dtype)
                self.assertTrue(torch.equal(out.cpu(), expected))
                self.assertEqual(out.dtype, dtype)
                self.assertTrue(torch.equal(xn.cpu(), original))
                self.assertEqual(int(torch_npu.get_npu_format(out)), 30)

    def test_npu_base_formats_are_exact(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        fn, _ = load()
        x = torch.randn(1, 16, 2, 7, 9)
        expected = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        for fmt in (2, 30):
            xn = torch_npu.npu_format_cast(x.to("npu"), fmt)
            out = fn(xn, (0, 1, 0, 1, 0, 0), "reflect")
            torch.npu.synchronize()
            self.assertTrue(torch.equal(out.cpu(), expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
