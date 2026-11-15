#!/usr/bin/env python3
"""Probe real RoPE tables and vision FP32 helper under explicit contexts.

This is an isolated NPU/context audit, not an assertion about the actual server
context until the focused runtime capture corroborates it. Source math is loaded
from inspected production classes; a protected FP32 candidate is separate.
"""
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
from typing import Any
import types
import torch
import torch_npu
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding,Qwen3VLVisionBlock

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'autocast_probe.json'
if OUT.exists():raise RuntimeError('Refuse context evidence overwrite')
torch.npu.set_device(0);torch.set_num_threads(16)
tree=ast.parse((ROOT/'source/encoder.py').read_text())
names={'MiniMaxH3Qwen3VLTextRotaryEmbedding','_apply_interleaved_mrope','_NpuBF16CompatibleLinear','_vision_fp32_enabled'}
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
inner=[]
def spy_linear(x,w,b=None):
 y=F.linear(x,w,b);inner.append({'input':str(x.dtype),'weight':str(w.dtype),'output':str(y.dtype),'autocast_enabled':torch.is_autocast_enabled('npu')});return y
ns={'torch':torch,'nn':nn,'F':types.SimpleNamespace(linear=spy_linear),'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_context_sources','exec'),ns)
config=Qwen3VLConfig.from_pretrained(str(MODEL));inputs=torch.load(CAP/'text_inputs.pt',weights_only=True)
actual_rope=torch.load(CAP/'text_rotary.pt',weights_only=True)
ref=Qwen3VLTextRotaryEmbedding(config.text_config)
with torch.inference_mode():hf_rope=ref(inputs['inputs_embeds'],inputs['positions'])
x=inputs['inputs_embeds'].to('npu');pos=inputs['positions'].to('npu');rope=ns['MiniMaxH3Qwen3VLTextRotaryEmbedding'](config.text_config)
r={'rope':[],'linear':[],'scope':'Isolated contexts on actual inputs; actual runtime context must be independently captured','npu_matmul_allow_hf32':torch.npu.matmul.allow_hf32}
def metric(a,b):
 a=a.detach().float().cpu();b=b.detach().float().cpu();e=a-b
 return {'bit_exact':bool(torch.equal(a,b)),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
contexts=[('default_cpu_noamp','cpu',None),('default_npu_noamp','npu',None),('default_cpu_bf16','cpu',torch.bfloat16),('default_cpu_fp16','cpu',torch.float16),('default_npu_bf16','npu',torch.bfloat16)]
try:
 with torch.inference_mode():
  for label,device,dtype in contexts:
   item={'label':label}
   try:
    with ExitStack() as stack:
     stack.enter_context(torch.device(device));stack.enter_context(torch.autocast('npu',dtype=dtype or torch.float16,enabled=dtype is not None))
     item['autocast_enabled']=torch.is_autocast_enabled('npu');item['autocast_dtype']=str(torch.get_autocast_dtype('npu'));item['default_device']=str(torch.get_default_device())
     got=rope(x,pos);torch.npu.synchronize()
     item['vs_hf']=[metric(a,b) for a,b in zip(got,hf_rope)];item['vs_actual_capture']=[metric(a,b) for a,b in zip(got,actual_rope)]
   except Exception as exc:item['failure']={'type':type(exc).__name__,'message':str(exc)}
   r['rope'].append(item);print(json.dumps(item),flush=True)
  # Protect the original frequency computation from inherited autocast. Explicit
  # CPU arange preserves inverse frequency precision even in a default-NPU scope.
  inv=ref.inv_freq.to('npu')
  def protected_rope():
   with torch.autocast('npu',enabled=False):
    freqs=(inv[None,None,:,None].float().expand(3,pos.shape[1],-1,1)@pos[:,:,None,:].float()).transpose(2,3)
    freqs=ns['_apply_interleaved_mrope'](freqs,rope.mrope_section);emb=torch.cat((freqs,freqs),dim=-1)
    return emb.cos().to(x.dtype),emb.sin().to(x.dtype)
  with torch.autocast('npu',dtype=torch.bfloat16,enabled=True):protected=protected_rope()
  r['protected_rope_vs_hf']=[metric(a,b) for a,b in zip(protected,hf_rope)]
  # Same actual block0 normalized CPU input and original qkv weights as round55.
  data=torch.load(Path('/tmp/h3_round55_vision01/block_00.pt'),weights_only=True)
  prefix='model.visual.blocks.0.';index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];state={}
  for shard in sorted({s for k,s in index.items() if k.startswith(prefix)}):
   with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
    for k,s in index.items():
     if s==shard and k.startswith(prefix):state[k[len(prefix):]]=f.get_tensor(k).bfloat16()
  teacher=Qwen3VLVisionBlock(config.vision_config).to(torch.bfloat16);teacher.load_state_dict(state,strict=True)
  hx=teacher.norm1(data['hidden']);hf=F.linear(hx.float(),teacher.attn.qkv.weight.float(),teacher.attn.qkv.bias.float()).bfloat16()
  model=ns['_NpuBF16CompatibleLinear'](config.vision_config.hidden_size,3*config.vision_config.hidden_size).to(torch.bfloat16)
  model.load_state_dict(teacher.attn.qkv.state_dict());model.to('npu');hx=hx.to('npu');os.environ['VLLM_OMNI_910A_VISION_FP32']='1'
  for label,device,dtype in contexts[:4]:
   inner.clear()
   with ExitStack() as stack:
    stack.enter_context(torch.device(device));stack.enter_context(torch.autocast('npu',dtype=dtype or torch.float16,enabled=dtype is not None))
    torch.npu.synchronize();t=time.perf_counter();got=model(hx);torch.npu.synchronize();dt=time.perf_counter()-t
    with torch.autocast('npu',enabled=False):good=F.linear(hx.float(),model.weight.float(),model.bias.float()).bfloat16()
   item={'label':label,'inner_linear':inner.copy(),'output_dtype':str(got.dtype),'one_call_s':dt,'vs_fp32_cpu':metric(got,hf),'protected_vs_fp32_cpu':metric(good,hf),'same_as_protected':metric(got,good)}
   r['linear'].append(item);print(json.dumps(item),flush=True)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
