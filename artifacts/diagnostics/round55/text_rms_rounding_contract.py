#!/usr/bin/env python3
"""Measure model-specific BF16 RMSNorm rounding contract before writing fused ops.

Uses actual layer0 input and actual layer24 output (the next layer25 input), plus
original per-layer gamma. This is CPU formula isolation, not a proof that it alone
causes layer50/video errors, and it does not change production normalization.
"""
import ast
import json
from pathlib import Path
import types
import torch
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

ROOT=Path('/tmp/h3_round55_vision01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'text_rms_rounding_contract.json'
if OUT.exists():raise RuntimeError('Refuse rounding-contract evidence overwrite')
torch.set_num_threads(16)
tree=ast.parse((ROOT/'source/norm.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RMSNorm')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_native')
ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_norm_fallback','exec'),ns)
config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config;index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
r={'cases':[],'scope':'CPU same-real-input normalization formula isolation; not a real NPU speed/layer50-cause or full-model quality pass','production_changed':False}
def metric(a,b):
 x=a.float();y=b.float();e=x-y
 return {'bit_exact':bool(torch.equal(a,b)),'max_abs_error':float(e.abs().max()),
         'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}
with torch.inference_mode():
 for layer,name in ((0,'text_inputs'),(25,'text_layer_24')):
  x=torch.load(CAP/(name+'.pt'),weights_only=True)
  if isinstance(x,dict):x=x['inputs_embeds']
  key=f'model.language_model.layers.{layer}.input_layernorm.weight'
  with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:gamma=f.get_tensor(key).bfloat16()
  norm=Qwen3VLTextRMSNorm(config.hidden_size,config.rms_norm_eps).to(torch.bfloat16);norm.weight.data.copy_(gamma)
  hf=norm(x);native=ns['forward_native'](types.SimpleNamespace(weight=gamma,variance_epsilon=config.rms_norm_eps),x)
  xf=x.float();normalized=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+config.rms_norm_eps)
  rounded_then_gamma=gamma*normalized.to(x.dtype)
  item={'normalization_layer':layer,'actual_input_capture':name,'native_fp32_gamma_then_cast_vs_hf':metric(native,hf),
        'bf16_normalized_then_gamma_vs_hf':metric(rounded_then_gamma,hf)}
  assert item['bf16_normalized_then_gamma_vs_hf']['bit_exact']
  r['cases'].append(item)
r['completed']=True;OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
