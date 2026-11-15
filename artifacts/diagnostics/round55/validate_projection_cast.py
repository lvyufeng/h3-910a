#!/usr/bin/env python3
"""Correct patch-projection comparison retaining the HF BF16 input cast.

The exploratory full_tower_ab extra projection variant used raw processor FP32
pixels; it is not a like-for-like precision comparison. This repeat explicitly
casts pixels to the stored projection dtype first, as the reference does.
"""
import ast
import json
from pathlib import Path
import time
import types
from typing import Any
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT=Path('/tmp/h3_round55_vision01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'projection_cast_validation.json'
if OUT.exists():raise RuntimeError('Refuse existing projection validation')
torch.npu.set_device(0);torch.set_num_threads(16)
r={'scope':'HF-equivalent BF16 pixel input cast and FP32 projection, with identical original weights/BF16 output; complete vision reference, not generation quality'}
def metric(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu();assert a.shape==b.shape;e=a-b
 return {'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'max_abs_error':float(e.abs().max()),
         'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
try:
 tree=ast.parse((ROOT/'candidate_encoder_unconditional_fp32.py').read_text());names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm','_NpuBF16CompatibleLinear'}
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
 ns={'torch':torch,'nn':nn,'F':F,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'vision_candidate','exec'),ns)
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;model=ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()
 wanted=set(dict(model.named_parameters()));state={};index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 for shard in sorted(set(index.values())):
  keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
  if not keys:continue
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).to(torch.bfloat16)
 model.load_state_dict(state,strict=True);inputs=torch.load(CAP/'encoder_input.pt',weights_only=True);hf=torch.load(CAP/'hf_vision_outputs.pt',weights_only=True)
 old_forward=model.patch_embed.forward
 width=config.in_channels*config.temporal_patch_size*config.patch_size*config.patch_size
 with torch.inference_mode():cpu_reference=model.patch_embed(inputs['pixel_values'])
 def forward(self,value):
  if value.device.type in ('npu','privateuseone') and self.proj.weight.dtype==torch.bfloat16:
   return F.linear(value.reshape(-1,width).to(self.proj.weight.dtype).float(),
                   self.proj.weight.reshape(config.hidden_size,-1).float(),self.proj.bias.float()).to(self.proj.weight.dtype)
  return old_forward(value)
 model.patch_embed.forward=types.MethodType(forward,model.patch_embed)
 with torch.inference_mode():assert torch.equal(model.patch_embed(inputs['pixel_values']),cpu_reference)
 r['cpu_patch_path_unchanged']=True
 model.to('npu');pixels=inputs['pixel_values'].to('npu');grid=inputs['image_grid_thw'].to('npu')
 with torch.inference_mode():
  patch=model.patch_embed(pixels);r['projection_vs_cpu_hf']=metric(patch,cpu_reference)
  torch.npu.synchronize();t=time.perf_counter();out,deep=model(pixels,grid);torch.npu.synchronize()
  r['full_tower_s']=time.perf_counter()-t
  r['merged_vs_cpu_hf']=metric(out,hf['merged']);r['deepstack_vs_cpu_hf']=[metric(a,b) for a,b in zip(deep,hf['deepstack'])]
 r['completed']=True
except BaseException as exc:
 r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
