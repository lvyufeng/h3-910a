#!/usr/bin/env python3
"""Teacher-force actual native layer49 input; isolate last-layer sub-operators.

References keep BF16 stored values/outputs, with explicitly labelled FP32 linear
oracle paths. One BF16 CPU last-layer repeat checks that this oracle substitution
alone does not explain a reported native discrepancy.
"""
import json
from pathlib import Path
import signal
import time
import types
import torch
from torch import nn
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer,Qwen3VLTextRMSNorm,Qwen3VLTextRotaryEmbedding

ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round56_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'last_layer_reference.json'
if OUT.exists():raise RuntimeError('Refuse last-layer evidence overwrite')
for name in ('layer49_input','layer49_output','layer49_attention','layer49_norm2','layer49_gate_up','layer49_down','text_input'):
 if not (CAP/(name+'.json')).exists():raise RuntimeError('Capture not ready: '+name)
torch.set_num_threads(16)
def interrupted(signum,frame):raise SystemExit('CPU last layer interrupted '+str(signum))
signal.signal(signal.SIGTERM,interrupted)
r={'cases':{},'suboperators':{},'scope':'Actual captured NPU input with original last-layer weights; teacher-forced CPU reference, not whole-encoder/video-quality validation'}
def load(name):return torch.load(CAP/(name+'.pt'),weights_only=True)
def metric(a,b):
 a=a.detach().float();b=b.detach().float();e=a-b
 return {'shape':list(a.shape),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'a_rms':float(a.square().mean().sqrt()),'b_rms':float(b.square().mean().sqrt()),'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}
try:
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config;config._attn_implementation='sdpa';config.use_cache=False
 model=Qwen3VLTextDecoderLayer(config,49).to(torch.bfloat16).eval()
 index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];prefix='model.language_model.layers.49.';state={}
 for shard in sorted({s for k,s in index.items() if k.startswith(prefix)}):
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k,s in index.items():
    if s==shard and k.startswith(prefix):state[k[len(prefix):]]=f.get_tensor(k).bfloat16()
 model.load_state_dict(state,strict=True)
 norms={m:m.forward for m in model.modules() if isinstance(m,Qwen3VLTextRMSNorm)}
 linears={m:m.forward for m in model.modules() if isinstance(m,nn.Linear)}
 def fp32_linear(self,x):return torch.nn.functional.linear(x.float(),self.weight.float(),self.bias.float() if self.bias is not None else None).to(x.dtype)
 def native_norm(self,x):
  xf=x.float();return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+self.variance_epsilon)*self.weight.float()).to(x.dtype)
 for m in linears:m.forward=types.MethodType(fp32_linear,m)
 actual_in=load('layer49_input');x=actual_in['args'][0];pe=actual_in['kwargs']['position_embeddings'];native=load('layer49_output')['output']
 full=load('text_input');hfpe=Qwen3VLTextRotaryEmbedding(config)(full['args'][0],full['args'][1])
 baseline=torch.load(ROOT/'cpu_last_hf.pt',weights_only=True)
 r['actual_native_input_vs_full_hf_input']=metric(x,baseline['hidden_states'])
 r['actual_context']=actual_in['context'];r['actual_rope_vs_cpu_hf']=[metric(a,b) for a,b in zip(pe,hfpe)]
 outputs={}
 with torch.inference_mode():
  for name,native_contract,table in [('native_norm_native_rope',True,pe),('hf_norm_native_rope',False,pe),('hf_norm_hf_rope',False,hfpe)]:
   for m,method in norms.items():m.forward=types.MethodType(native_norm,m) if native_contract else method
   t=time.perf_counter();out=model(x,position_embeddings=table,use_cache=False)
   outputs[name]=out;r['cases'][name]={'wall_s':time.perf_counter()-t,'output_vs_native':metric(out,native)}
  for m,method in norms.items():m.forward=types.MethodType(native_norm,m)
  for m,method in linears.items():m.forward=method
  t=time.perf_counter();bf16=model(x,position_embeddings=pe,use_cache=False)
  r['cases']['native_norm_native_rope_cpu_bf16_linear']={'wall_s':time.perf_counter()-t,'output_vs_native':metric(bf16,native),'vs_fp32_linear_oracle':metric(bf16,outputs['native_norm_native_rope'])}
  for m in linears:m.forward=types.MethodType(fp32_linear,m)
  # Every sub-operator below receives its actual captured native input.
  attn=load('layer49_attention');attn_input=attn['args'][0]
  for name,native_contract in [('native_qk_norm',True),('hf_qk_norm',False)]:
   for m,method in norms.items():m.forward=types.MethodType(native_norm,m) if native_contract else method
   y,_=model.self_attn(attn_input,position_embeddings=pe,attention_mask=None)
   r['suboperators']['attention_'+name]=metric(y,attn['output'])
  mlp_input=load('layer49_norm2')['output'][0]
  gate=model.mlp.gate_proj(mlp_input);up=model.mlp.up_proj(mlp_input);activation=model.mlp.act_fn(gate)*up
  mlp=model.mlp.down_proj(activation);actual_gate=load('layer49_gate_up')['output'];actual_down=load('layer49_down')['output']
  size=config.intermediate_size//8;rank0=torch.cat((gate[...,:size],up[...,:size]),dim=-1)
  r['suboperators']['packed_gate_up_rank0_vs_full_fp32_reference']=metric(actual_gate,rank0)
  r['suboperators']['full_down_vs_actual_reduced_down']=metric(actual_down,mlp)
  local=[]
  for rank in range(8):
   start=rank*size;end=start+size
   local.append(torch.nn.functional.linear(activation[...,start:end].float(),model.mlp.down_proj.weight[:,start:end].float()).bfloat16())
  partial_sum=torch.stack([v.float() for v in local]).sum(0).bfloat16()
  r['suboperators']['fp32_sum_of_bf16_partials_vs_actual_down']=metric(actual_down,partial_sum)
  r['suboperators']['bf16_partials_fp32_sum_vs_unsharded_reference']=metric(partial_sum,mlp)
  target=ROOT/'last_reference_outputs.pt'
  if target.exists():raise RuntimeError('Refuse output tensors overwrite')
  with target.open('xb') as f:torch.save(outputs,f)
 r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('Evidence:',OUT,flush=True)
