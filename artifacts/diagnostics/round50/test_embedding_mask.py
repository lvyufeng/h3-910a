#!/usr/bin/env python3
"""CPU and 910A regressions for the exact production TP embedding mask repair.

Load only the inspected class AST, avoiding vLLM/model loading. Eight mock TP
ranks exercise local-shard masking and sum reconstruction; the real worker's
collective remains unchanged. The old Boolean indexing is evaluated on CPU only.
"""
import ast
import copy
from pathlib import Path
import types
import unittest
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent


def load_class(path):
    tree = ast.parse(Path(path).read_text())
    names = {"_tp_range", "MiniMaxH3Qwen3VLVocabParallelEmbedding"}
    nodes = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in names]
    assert len(nodes) == 2
    scope = {"torch": torch, "nn": nn, "F": F, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["MiniMaxH3Qwen3VLVocabParallelEmbedding"]


def load_token_types(path):
    tree = ast.parse(Path(path).read_text())
    klass = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "MiniMaxH3Qwen3VLEncoder")
    fn = copy.deepcopy(next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == "_encode"))
    body = []
    for node in fn.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "inputs_embeds" for t in node.targets):
            break
        body.append(node)
    fn.body = body + [ast.Return(value=ast.Name(id="mm_types", ctx=ast.Load()))]
    fn.decorator_list = []
    scope = {"torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["_encode"]


class Group:
    def __init__(self, rank, world):
        self.rank_in_group = rank
        self.world_size = world
        self.reductions = 0

    def all_reduce(self, output):
        self.reductions += 1
        return output


class EmbeddingMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_class(ROOT / "source/encoder.py")
        cls.new = load_class(ROOT / "candidate_encoder.py")

    def check(self, inputs, dtype, device="cpu", world=8):
        weight = torch.arange(32 * 16, dtype=dtype).reshape(32, 16) / 32
        expected = F.embedding(inputs.cpu(), weight)
        outputs = []
        for rank in range(world):
            group = Group(rank, world)
            candidate = self.new(group, 32, 16, dtype).to(device)
            candidate.weight.data.copy_(weight[rank * (32 // world):(rank + 1) * (32 // world)].to(device))
            x = inputs.to(device)
            before = x.clone()
            output = candidate(x)
            if device == "cpu":
                old = self.old(Group(rank, world), 32, 16, dtype)
                old.weight.data.copy_(candidate.weight)
                self.assertTrue(torch.equal(output, old(inputs)))
            self.assertTrue(torch.equal(x, before))
            self.assertEqual(group.reductions, int(world > 1))
            outputs.append(output.detach().cpu())
        merged = torch.stack(outputs).sum(0)
        self.assertTrue(torch.equal(merged, expected))

    def test_all_partition_boundaries_fp16(self):
        self.check(torch.arange(32).reshape(2, 16), torch.float16)

    def test_all_partition_boundaries_fp32(self):
        self.check(torch.arange(32).reshape(2, 16), torch.float32)

    def test_single_rank_fast_path_unchanged(self):
        self.check(torch.tensor([[0, 31, 7, 7]]), torch.float16, world=1)

    def test_long_repeated_image_text_tokens(self):
        self.check(torch.arange(32).repeat(32).reshape(1, 1024), torch.float16)

    def test_empty_and_all_out_of_local_shard(self):
        self.check(torch.empty((1, 0), dtype=torch.long), torch.float32)
        self.check(torch.tensor([[31, 31, 31]]), torch.float16)

    def test_invalid_global_ids_retain_original_zero_semantics(self):
        x = torch.tensor([[-1, 32, 128]])
        new = self.new(Group(3, 8), 32, 16, torch.float32)
        old = self.old(Group(3, 8), 32, 16, torch.float32)
        old.weight.data.copy_(new.weight)
        self.assertTrue(torch.equal(new(x), old(x)))
        self.assertTrue(torch.equal(new(x), torch.zeros(1, 3, 16)))

    def test_token_type_masking_cpu_and_override_priority(self):
        old = load_token_types(ROOT / "source/encoder.py")
        new = load_token_types(ROOT / "candidate_encoder.py")
        x = torch.tensor([0, 31, 7, 7, 31, 0])
        for image_id, video_id in ((7, 31), (7, 7), (100, 101)):
            ctx = types.SimpleNamespace(device_target="cpu", image_token_id=image_id, video_token_id=video_id)
            self.assertTrue(torch.equal(new(ctx, x), old(ctx, x)))
            self.assertEqual(new(ctx, x).dtype, torch.int32)

    def test_910a_int32_token_type_masking(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        x = torch.tensor([0, 31, 7, 7, 31, 0])
        ctx = types.SimpleNamespace(device_target="npu", image_token_id=7, video_token_id=31)
        result = load_token_types(ROOT / "candidate_encoder.py")(ctx, x)
        self.assertTrue(torch.equal(result.cpu(), torch.tensor([[0, 2, 1, 1, 2, 0]], dtype=torch.int32)))
        torch.npu.synchronize()

    def test_910a_int64_masking_and_embedding(self):
        try:
            import torch_npu  # noqa
        except ImportError:
            self.skipTest("torch_npu unavailable")
        torch.npu.set_device(0)
        self.check(torch.arange(32).repeat(32).reshape(1, 1024), torch.float16, device="npu")
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
