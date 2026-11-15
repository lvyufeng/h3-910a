#!/usr/bin/env python3
"""Verify BF16-linear compatibility formula and unchanged non-NPU paths."""
import ast
import json
from pathlib import Path
import types
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'candidate_encoder.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('_NpuBF16CompatibleLinear','_vision_fp32_enabled')]
ns={'torch':torch,'nn':nn,'F':F};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'candidate_linear','exec'),ns)
Compat=ns['_NpuBF16CompatibleLinear']
count=0
for bias in (False,True):
 for dtype in (torch.float16,torch.bfloat16,torch.float32):
  torch.manual_seed(55)
  old=nn.Linear(32,19,bias=bias).to(dtype)
  new=Compat(32,19,bias=bias).to(dtype);new.load_state_dict(old.state_dict(),strict=True)
  assert list(old.state_dict())==list(new.state_dict())
  x=torch.randn(3,7,32).to(dtype)
  with torch.inference_mode():assert torch.equal(new(x),old(x))
  count+=1
 # Drive only the narrow FP32 formula branch with CPU tensors. Real NPU math is
 # independently checked by the full-tower A/B, so this does not fake hardware.
 new=Compat(32,19,bias=bias).to(torch.bfloat16)
 x=torch.randn(7,32).bfloat16()
 mock=types.SimpleNamespace(device=types.SimpleNamespace(type='npu'),dtype=x.dtype,float=lambda:x.float())
 with patch.dict('os.environ',{'VLLM_OMNI_910A_VISION_FP32':'1'}),torch.inference_mode():
  got=new.forward(mock)
  expected=F.linear(x.float(),new.weight.float(),new.bias.float() if bias else None).bfloat16()
 assert torch.equal(got,expected) and got.dtype==torch.bfloat16
 count+=1
report={'passed':True,'formula_and_cpu_cases':count,'cpu_fp16_bf16_fp32_bit_exact_unchanged':True,
        'parameter_names_and_dtypes_preserved':True,'fp32_formula_then_single_bf16_cast_exact_on_cpu':True,
        'scope':'Formula/non-NPU regression; real NPU operation is validated separately, not simulated by this test'}
(ROOT/'linear_regressions_optin.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
