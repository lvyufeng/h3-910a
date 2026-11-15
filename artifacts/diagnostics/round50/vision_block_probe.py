#!/usr/bin/env python3
"""Smoke the exact patched vision block before paying for an 8-rank engine boot.

Keeps BF16 storage/activations. Loads only selected source AST definitions; no
model assets are loaded. CPU comparison is diagnostic, not a model-quality claim.
"""
import ast
import copy
import json
import os
from pathlib import Path
import types
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

ROOT = Path("/tmp/h3_round50_norm_validation01")
OUT = ROOT / os.environ.get("H3_VISION_PROBE_OUT", "vision_block_probe01.json")
if OUT.exists():
    raise RuntimeError("Refuse to overwrite existing block probe")
names = {"_NpuBF16CompatibleLayerNorm", "MiniMaxH3Qwen3VLVisionMLP", "_rotate_half", "_vision_sdpa_bf16_compatible",
         "_apply_rotary_pos_emb_vision", "MiniMaxH3Qwen3VLVisionAttention", "MiniMaxH3Qwen3VLVisionBlock"}
tree = ast.parse((ROOT / "candidate_encoder.py").read_text())
nodes = [x for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef)) and x.name in names]
assert len(nodes) == len(names)
scope = {"torch": torch, "nn": nn, "F": F, "Any": Any}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "real-vision-block", "exec"), scope)
cfg = types.SimpleNamespace(hidden_size=1152, intermediate_size=4304, num_heads=16)
torch.manual_seed(50)
model = scope["MiniMaxH3Qwen3VLVisionBlock"](cfg).to(torch.bfloat16).eval()
x = torch.randn(128, 1152, dtype=torch.bfloat16)
cu = torch.tensor([0, 64, 128], dtype=torch.int32)
cos = torch.ones(128, 72)
sin = torch.zeros_like(cos)
result = {"source": str(ROOT / "candidate_encoder.py"), "shape": list(x.shape), "dtype": "bfloat16"}
try:
    with torch.inference_mode():
        ref = model(x, cu, (cos, sin))
        torch.npu.set_device(0)
        model = model.to("npu")
        out = model(x.to("npu"), cu.to("npu"), (cos.to("npu"), sin.to("npu")))
        torch.npu.synchronize()
        out_cpu = out.cpu()
    err = (out_cpu.float() - ref.float()).abs()
    result.update({"finite": bool(torch.isfinite(out_cpu).all()), "output_shape": list(out.shape),
                   "output_dtype": str(out.dtype), "max_abs_diff_vs_cpu_bf16": err.max().item(),
                   "relative_rms_vs_cpu_bf16": (err.square().mean().sqrt() / ref.float().square().mean().sqrt()).item()})
    assert result["finite"] and out.dtype == torch.bfloat16 and out.shape == x.shape
    assert result["relative_rms_vs_cpu_bf16"] < .005, result
except BaseException as exc:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
