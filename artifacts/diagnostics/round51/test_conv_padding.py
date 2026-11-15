#!/usr/bin/env python3
"""CPU/NPU regressions for exact local VAE convolution padding repairs."""
import ast
from pathlib import Path
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def load(path, candidate):
    names = {"BaseConv3d", "SpatialParallelConv3d"}
    if candidate:
        names.add("_pad_base_format")
    tree = ast.parse(Path(path).read_text())
    nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
    scope = {"torch": torch, "nn": nn, "F": F,
             "get_parallel_state": lambda: {"sp_rank": 0, "sp_size": 1, "sp_process_group": None},
             "exchange_borders": lambda x, *args, **kwargs: x}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ConvPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load(ROOT / "source/cached_conv.py", False)
        cls.new = load(ROOT / "candidate_conv.py", True)

    def test_cpu_zero_padding_bit_exact(self):
        for dtype in (torch.float32, torch.float16):
            for mode in ("zeros", "reflect", "replicate"):
                old = self.old["BaseConv3d"](3, 4, 1, padding=0, padding_mode=mode).to(dtype)
                new = self.new["BaseConv3d"](3, 4, 1, padding=0, padding_mode=mode).to(dtype)
                new.load_state_dict(old.state_dict(), strict=True)
                x = torch.randn(1, 3, 1, 8, 10, dtype=dtype)
                self.assertTrue(torch.equal(new(x), old(x)))
                self.assertEqual(set(new.state_dict()), {"weight", "bias"})

    def test_cpu_spatial_and_temporal_padding_unchanged(self):
        for causal in (True, False):
            for d in (1, 3):
                old = self.old["BaseConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect", causal=causal)
                new = self.new["BaseConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect", causal=causal)
                new.load_state_dict(old.state_dict(), strict=True)
                x = torch.randn(1, 3, d, 8, 10)
                self.assertTrue(torch.equal(new._apply_padding(x), old._apply_padding(x)))
                self.assertTrue(torch.equal(new(x), old(x)))

    def test_cpu_non_spatial_branch_preserves_behavior(self):
        old = self.old["SpatialParallelConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect")
        new = self.new["SpatialParallelConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect")
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(1, 3, 3, 8, 10)
        self.assertTrue(torch.equal(new(x), old(x)))

    def test_cpu_spatial_boundary_padding_unchanged(self):
        for dim in (-1, -2):
            old = self.old["SpatialParallelConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect")
            new = self.new["SpatialParallelConv3d"](3, 4, 3, padding=(1, 1, 1), padding_mode="reflect")
            old.spatial_parallel = new.spatial_parallel = True
            old.chunk_dim = new.chunk_dim = dim
            x = torch.randn(1, 3, 3, 8, 10)
            self.assertTrue(torch.equal(new._apply_padding(x), old._apply_padding(x)))

    def test_npu_zero_padding_shortcut_runs_on_packed_input(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        for dtype in (torch.float32, torch.float16):
            layer = self.new["BaseConv3d"](16, 32, 1, padding=0, padding_mode="reflect").to(dtype)
            x = torch.randn(1, 16, 1, 8, 10, dtype=dtype)
            expected = layer(x).detach().float()
            xn = torch_npu.npu_format_cast(x.to("npu"), 32)
            layer = layer.to("npu")
            out = layer(xn)
            torch.npu.synchronize()
            got = out.detach().cpu().float()
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(got.shape, expected.shape)
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, expected, atol=.005, rtol=.005)

    def test_npu_positive_padding_values_are_exact_before_conv(self):
        try:
            import torch_npu
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        for dtype in (torch.float32, torch.float16):
            layer = self.new["BaseConv3d"](16, 16, 3, padding=(1, 1, 1), padding_mode="reflect").to(dtype)
            x = torch.randn(1, 16, 3, 8, 10, dtype=dtype)
            expected = layer._apply_padding(x)
            xn = torch_npu.npu_format_cast(x.to("npu"), 32)
            actual = layer._apply_padding(xn)
            torch.npu.synchronize()
            self.assertTrue(torch.equal(actual.cpu(), expected))
            self.assertEqual(actual.dtype, dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
