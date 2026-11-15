#!/usr/bin/env python3
"""Separate actual RoPE inverse-frequency, outer-product, layout and trig errors."""
import ast
import json
from pathlib import Path
from typing import Any
import torch
import torch_npu
from torch import nn
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round54_capture01');MODEL='/home/mseco/models/MiniMax-H3/FL2VA/text_encoder';OUT=ROOT/'rope_stages.json'
if OUT.exists():raise RuntimeError('Refuse RoPE stage evidence overwrite')
torch.npu.set_device(0);torch.set_num_threads(16)
config=Qwen3VLConfig.from_pretrained(MODEL).text_config;inp=torch.load(CAP/'text_inputs.pt',weights_only=True)
ref=Qwen3VLTextRotaryEmbedding(config);pos=inp['positions'].to('npu');x=inp['inputs_embeds'].to('npu')
tree=ast.parse((ROOT/'source/encoder.py').read_text());nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('MiniMaxH3Qwen3VLTextRotaryEmbedding','_apply_interleaved_mrope')]
ns={'torch':torch,'nn':nn,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_rope_source','exec'),ns)
rope=ns['MiniMaxH3Qwen3VLTextRotaryEmbedding'](config)
r={'stages':{},'scope':'Actual positions with isolated NPU frequency/layout/trig paths, not complete encoder/quality validation'}
def metric(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu();e=a-b
 return {'bit_exact':bool(torch.equal(a,b)),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
try:
 with torch.inference_mode(),torch.autocast('npu',enabled=False):
  inv=rope._get_inv_freq(pos.device);r['inverse_frequency']=metric(inv,ref.inv_freq)
  ac=ref.inv_freq[None,None,:,None].float().expand(3,pos.shape[1],-1,1);bc=inp['positions'][:,:,None,:].float()
  cpu_raw=(ac@bc).transpose(2,3)
  a=inv[None,None,:,None].float().expand(3,pos.shape[1],-1,1);b=pos[:,:,None,:].float()
  raw=(a@b).transpose(2,3);raw_contig=(a.contiguous()@b.contiguous()).transpose(2,3).contiguous()
  raw_mul=inv[None,None,None,:].float()*pos[:,:,:,None].float()
  r['raw_matmul']=metric(raw,cpu_raw);r['raw_matmul_contiguous_inputs']=metric(raw_contig,cpu_raw);r['raw_elementwise_outer_product']=metric(raw_mul,cpu_raw)
  cpu_freq=ref.apply_interleaved_mrope(cpu_raw.clone(),ref.mrope_section)
  hf=ref(inp['inputs_embeds'],inp['positions'])
  for label,source in [('native_layout',raw),('contiguous_matmul',raw_contig),('contiguous_elementwise',raw_mul)]:
   freq=ns['_apply_interleaved_mrope'](source.clone(),rope.mrope_section);emb=torch.cat((freq,freq),dim=-1)
   r['stages'][label]={'frequency':metric(freq,cpu_freq),'tables':[metric(t,u) for t,u in zip((emb.cos().to(x.dtype),emb.sin().to(x.dtype)),hf)]}
  # Avoid strided mutation using explicit modality masks on a contiguous tensor.
  idx=torch.arange(cpu_freq.shape[-1],device='npu');mh=(idx%3==1)&(idx<rope.mrope_section[1]*3);mw=(idx%3==2)&(idx<rope.mrope_section[2]*3)
  freq=torch.where(mh,raw_mul[1].clone(),torch.where(mw,raw_mul[2].clone(),raw_mul[0].clone()))
  emb=torch.cat((freq,freq),dim=-1)
  r['stages']['elementwise_where']={'frequency':metric(freq,cpu_freq),'tables':[metric(t,u) for t,u in zip((emb.cos().to(x.dtype),emb.sin().to(x.dtype)),hf)]}
  emb=torch.cat((cpu_freq,cpu_freq),dim=-1).to('npu')
  r['cpu_exact_frequency_npu_trig']=[metric(t,u) for t,u in zip((emb.cos().to(x.dtype),emb.sin().to(x.dtype)),hf)]
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
