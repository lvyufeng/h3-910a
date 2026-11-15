#!/usr/bin/env python3
"""Full real vision tower A/B for FP32 compute with unchanged BF16 storage/I/O.

Baseline and candidate use identical real processor tensors, parameters and
local NPU math settings. Includes casts/allocations and cold/warm separation.
An additional patch-projection candidate tests equivalent single-patch FP32
linear math without mutating any production model or device-global policy.
"""
import ast
import json
from pathlib import Path
import statistics
import time
import types
from typing import Any
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig

ROOT=Path('/tmp/h3_round55_vision01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'full_tower_ab.json'
if OUT.exists():raise RuntimeError('Refuse previous full-tower validation')
torch.npu.set_device(0);torch.set_num_threads(16)
r={'cases':{},'timings':[],'npu_matmul_allow_hf32':torch.npu.matmul.allow_hf32,'npu_conv_allow_hf32':torch.npu.conv.allow_hf32,
   'scope':'Same-input/operator/config complete vision tower A/B, no whole-text-encoder or generation-quality proof'}

def build(path,config):
 tree=ast.parse(path.read_text());names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm','_NpuBF16CompatibleLinear'}
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
 ns={'torch':torch,'nn':nn,'F':F,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns)
 return ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()

def metric(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu();assert a.shape==b.shape;e=a-b;finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
 return {'shape':list(a.shape),'finite':finite,'max_abs_error':float(e.abs().max()) if finite else None,
         'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12)) if finite else None}

def measure(call):
 torch.npu.synchronize();t=time.perf_counter();out=call();torch.npu.synchronize();return out,time.perf_counter()-t
try:
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config
 old=build(ROOT/'source/encoder.py',config);new=build(ROOT/'candidate_encoder_unconditional_fp32.py',config)
 wanted=set(dict(old.named_parameters()));assert wanted==set(dict(new.named_parameters()))
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
 for shard in sorted(set(index.values())):
  keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
  if not keys:continue
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).to(torch.bfloat16)
 old.load_state_dict(state,strict=True);new.load_state_dict(state,strict=True)
 assert all(torch.equal(old.state_dict()[k],new.state_dict()[k]) for k in wanted)
 r['same_bf16_parameter_state']=True;r['parameter_tensors']=len(wanted)
 inputs=torch.load(CAP/'encoder_input.pt',weights_only=True);teacher=torch.load(CAP/'hf_vision_outputs.pt',weights_only=True)
 with torch.inference_mode():
  # Non-NPU interpolation and linear execution remain the old path.
  assert torch.equal(old.fast_pos_embed_interpolate(inputs['image_grid_thw']),new.fast_pos_embed_interpolate(inputs['image_grid_thw']))
  cpu_patch=old.patch_embed(inputs['pixel_values'])
  r['cpu_path_interpolation_unchanged']=True
  old.to('npu');new.to('npu');pixels=inputs['pixel_values'].to('npu');grid=inputs['image_grid_thw'].to('npu')
  old_patch,dt=measure(lambda:old.patch_embed(pixels));r['patch_projection_old_vs_hf']=metric(old_patch,cpu_patch)
  patch_width=config.in_channels*config.temporal_patch_size*config.patch_size*config.patch_size
  def patch_fp32(self,value):
   if value.device.type in ('npu','privateuseone') and self.proj.weight.dtype==torch.bfloat16:
    return F.linear(value.reshape(-1,patch_width).float(),self.proj.weight.reshape(config.hidden_size,-1).float(),self.proj.bias.float()).bfloat16()
   return old_patch_method(value)
  # This is only a separate validation variant, not the deployed candidate yet.
  projected=F.linear(pixels.reshape(-1,patch_width).float(),new.patch_embed.proj.weight.reshape(config.hidden_size,-1).float(),new.patch_embed.proj.bias.float()).bfloat16()
  r['patch_projection_fp32_linear_vs_hf']=metric(projected,cpu_patch)
  models={'baseline':old,'fp32_linear_and_interpolation':new}
  for label,model in models.items():
   output,dt=measure(lambda:model(pixels,grid))
   r['cases'][label]={'cold_s':dt,'merged':metric(output[0],teacher['merged']),
                      'deepstack':[metric(a,b) for a,b in zip(output[1],teacher['deepstack'])]}
   assert output[0].dtype==torch.bfloat16
  # Interleaved order and reversed order, two full calls each. Timings include
  # conversion/temporary buffers; these are isolated full tower, not E2E latency.
  for order in (('baseline','fp32_linear_and_interpolation'),('fp32_linear_and_interpolation','baseline')):
   for label in order:
    output,dt=measure(lambda:models[label](pixels,grid));r['timings'].append({'variant':label,'warm_s':dt})
  old_patch_method=new.patch_embed.forward
  new.patch_embed.forward=types.MethodType(patch_fp32,new.patch_embed)
  out,dt=measure(lambda:new(pixels,grid))
  r['cases']['plus_fp32_patch_projection']={'warm_s':dt,'merged':metric(out[0],teacher['merged']),
                                         'deepstack':[metric(a,b) for a,b in zip(out[1],teacher['deepstack'])]}
  if r['cases']['plus_fp32_patch_projection']['merged']['relative_rms']>r['cases']['fp32_linear_and_interpolation']['merged']['relative_rms']:
   r['patch_projection_recommended']=False
  else:r['patch_projection_recommended']=True
  r['warm_mean_s']={label:statistics.mean(x['warm_s'] for x in r['timings'] if x['variant']==label) for label in models}
  r['speedup']=r['warm_mean_s']['baseline']/r['warm_mean_s']['fp32_linear_and_interpolation']
  r['cpu_hf_sample_scope']='Two complete real 48x84 images, full merged and deepstack tensors, BF16 stored parameters/output'
  r['completed']=True
except BaseException as exc:
 r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True);print('Evidence:',OUT,flush=True)
