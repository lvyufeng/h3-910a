#!/usr/bin/env python3
"""Verify default-off behavior and that opt-in executes the measured vision path."""
import ast
import json
import os
from pathlib import Path
import time
from typing import Any
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT=Path('/tmp/h3_round55_vision01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'optin_validation.json'
if OUT.exists():raise RuntimeError('Refuse existing opt-in validation')
torch.npu.set_device(0);torch.set_num_threads(16)
def build(path,config):
 tree=ast.parse(path.read_text());names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm','_NpuBF16CompatibleLinear','_vision_fp32_enabled'}
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
 ns={'torch':torch,'nn':nn,'F':F,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
 return ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()
def metric(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu();e=a-b
 return {'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'max_abs_error':float(e.abs().max()),
         'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
r={'scope':'Default-off operator behavior and opt-in full real tower; final video quality is separately unverified'}
try:
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;old=build(ROOT/'source/encoder.py',config);new=build(ROOT/'candidate_encoder.py',config)
 wanted=set(dict(old.named_parameters()));assert wanted==set(dict(new.named_parameters()));state={};index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 for shard in sorted(set(index.values())):
  keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
  if not keys:continue
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).bfloat16()
 old.load_state_dict(state);new.load_state_dict(state);old.to('npu');new.to('npu')
 inp=torch.load(CAP/'encoder_input.pt',weights_only=True);hf=torch.load(CAP/'hf_vision_outputs.pt',weights_only=True)
 pixels=inp['pixel_values'].to('npu');grid=inp['image_grid_thw'].to('npu');x=torch.randn(7,1152,dtype=torch.bfloat16,device='npu')
 before=(torch.npu.matmul.allow_hf32,torch.npu.conv.allow_hf32)
 with torch.inference_mode():
  os.environ.pop('VLLM_OMNI_910A_VISION_FP32',None)
  assert torch.equal(old.blocks[0].attn.qkv(x),new.blocks[0].attn.qkv(x))
  assert torch.equal(old.fast_pos_embed_interpolate(grid),new.fast_pos_embed_interpolate(grid))
  r['default_disabled_npu_linear_and_interpolation_exact']=True
  os.environ['VLLM_OMNI_910A_VISION_FP32']='1'
  new(pixels,grid);torch.npu.synchronize();t=time.perf_counter();output,deep=new(pixels,grid);torch.npu.synchronize()
  r['optin_warm_s']=time.perf_counter()-t;r['merged']=metric(output,hf['merged']);r['deepstack']=[metric(a,b) for a,b in zip(deep,hf['deepstack'])]
  previous=json.loads((ROOT/'full_tower_ab.json').read_text())['cases']['fp32_linear_and_interpolation']
  assert abs(r['merged']['relative_rms']-previous['merged']['relative_rms'])<1e-6
  assert output.dtype==torch.bfloat16 and all(p.dtype==torch.bfloat16 for p in new.parameters())
  assert (torch.npu.matmul.allow_hf32,torch.npu.conv.allow_hf32)==before
  r['same_as_measured_unconditional_path']=True;r['bf16_parameters_and_output_preserved']=True;r['global_device_precision_flags_unchanged']=True
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
