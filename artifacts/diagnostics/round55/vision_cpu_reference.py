#!/usr/bin/env python3
"""Corrected native CPU vision tower plus matched-input HF block references.

Uses the real round-54 processor tensors and original weights. Selected block
inputs/outputs are saved so an NPU replay does not confound accumulated errors.
"""
import ast
import json
from pathlib import Path
import signal
import time
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

ROOT=Path('/tmp/h3_round55_vision01');CAP=Path('/tmp/h3_round54_capture01')
CODE=ROOT/'source/encoder.py';MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'cpu_reference.json'
if OUT.exists():raise RuntimeError('Refuse existing reference receipt')
torch.set_num_threads(16)
report={'blocks':[],'scope':'Same real native CPU block inputs passed to installed HF blocks; corrected interpolation isolates accumulated vs per-block numerical differences'}
def interrupted(signum,frame):raise SystemExit('reference interrupted '+str(signum))
signal.signal(signal.SIGTERM,interrupted)

def load_native(path):
 tree=ast.parse(path.read_text());names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm'}
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
 ns={'torch':torch,'nn':nn,'F':F,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),ns);return ns

def metrics(a,b):
 a=a.detach().float();b=b.detach().float();assert a.shape==b.shape;d=a-b
 return {'shape':list(a.shape),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
         'bit_exact':bool(torch.equal(a,b)),'max_abs_error':float(d.abs().max()),
         'relative_rms':float(d.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
try:
 ns=load_native(CODE);config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;config._attn_implementation='sdpa'
 inputs=torch.load(CAP/'encoder_input.pt',weights_only=True);previous_hf=torch.load(CAP/'hf_vision_outputs.pt',weights_only=True)
 model=ns['MiniMaxH3Qwen3VLVisionModel'](config).to(torch.bfloat16).eval()
 wanted=set(dict(model.named_parameters()));state={};index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 for shard in sorted(set(index.values())):
  keys=[k for k,s in index.items() if s==shard and k.startswith('model.visual.') and k[len('model.visual.'):] in wanted]
  if not keys:continue
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k in keys:state[k[len('model.visual.'):]]=f.get_tensor(k).to(torch.bfloat16)
 model.load_state_dict(state,strict=True)
 # Exercise the known CPU FP32 interpolation prototype, without modifying
 # production's CPU path. We already separately checked its table against HF.
 tree=ast.parse(CODE.read_text());cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='MiniMaxH3Qwen3VLVisionModel')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fast_pos_embed_interpolate')
 src=ast.unparse(fn).replace('dtype=self.pos_embed.weight.dtype','dtype=torch.float32').replace('self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]','self.pos_embed(idx_tensor).float() * weight_tensor[:, :, None]').replace('pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]','pos_embeds.sum(dim=0)').replace('return torch.cat(patch_pos_embeds_permute)','return torch.cat(patch_pos_embeds_permute).to(self.pos_embed.weight.dtype)')
 env=dict(ns);exec(src,env)
 import types
 model.fast_pos_embed_interpolate=types.MethodType(env['fast_pos_embed_interpolate'],model)
 selected=(0,8,16,24,26);captures={}
 def pre(index):
  def capture(module,args,kwargs):
   captures[index]={'hidden':args[0].detach().clone(),'cu_seqlens':kwargs['cu_seqlens'].detach().clone(),
                    'position_embeddings':tuple(x.detach().clone() for x in kwargs['position_embeddings'])}
  return capture
 def post(index):
  def compare(module,args,out):
   payload=captures.pop(index);payload['native_cpu_output']=out.detach().clone()
   teacher=Qwen3VLVisionBlock(config).to(torch.bfloat16).eval();teacher.load_state_dict(module.state_dict(),strict=True)
   with torch.inference_mode():hf=teacher(payload['hidden'],cu_seqlens=payload['cu_seqlens'],position_embeddings=payload['position_embeddings'])
   payload['hf_cpu_output']=hf.detach().clone()
   target=ROOT/f'block_{index:02d}.pt'
   if target.exists():raise RuntimeError('Refuse existing matched block')
   with target.open('xb') as f:torch.save(payload,f)
   item={'block':index,'native_cpu_vs_hf_same_input':metrics(out,hf),'capture_file':target.name,'capture_bytes':target.stat().st_size}
   report['blocks'].append(item);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
  return compare
 for i in selected:
  model.blocks[i].register_forward_pre_hook(pre(i),with_kwargs=True);model.blocks[i].register_forward_hook(post(i))
 start=time.perf_counter()
 with torch.inference_mode():merged,deep=model(inputs['pixel_values'],inputs['image_grid_thw'])
 report['cpu_corrected_tower_wall_s']=time.perf_counter()-start
 report['corrected_native_cpu_vs_cached_hf_merged']=metrics(merged,previous_hf['merged'])
 report['corrected_native_cpu_vs_cached_hf_deepstack']=[metrics(a,b) for a,b in zip(deep,previous_hf['deepstack'])]
 report['completed']=True
 print(json.dumps(report,indent=2),flush=True)
except BaseException as exc:
 report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',OUT,flush=True)
