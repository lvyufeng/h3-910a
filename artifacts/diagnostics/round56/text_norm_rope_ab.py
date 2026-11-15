#!/usr/bin/env python3
"""Complete 50-layer CPU norm/RoPE ablations on the same real old capture.

All linears use explicit FP32 compute then BF16 output, as round55's oracle.
Norm and RoPE changes are independent. Late input/output captures are bounded
and reused for exact-input NPU sub-operator verification, not video quality.
"""
import argparse
import json
from pathlib import Path
import signal
import time
import types
import torch
from torch import nn
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel,Qwen3VLTextRotaryEmbedding,Qwen3VLTextRMSNorm

p=argparse.ArgumentParser();p.add_argument('--modes',required=True);a=p.parse_args()
modes=a.modes.split(',');assert modes and len(set(modes))==len(modes)
assert all(m in ('hf','norm-native','rope-native','both-native') for m in modes)
ROOT=Path('/tmp/h3_round56_audit01');CAP=Path('/tmp/h3_round54_capture01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
torch.set_num_threads(16)
for mode in modes:
 if (ROOT/f'cpu_ab_{mode}.json').exists():raise RuntimeError('Refuse ablation evidence overwrite: '+mode)
def interrupted(signum,frame):raise SystemExit('CPU reference interrupted '+str(signum))
signal.signal(signal.SIGTERM,interrupted)

def metric(a,b):
 a=a.detach().float();b=b.detach().float();assert a.shape==b.shape;e=a-b
 return {'shape':list(a.shape),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
         'reference_rms':float(b.square().mean().sqrt()),'native_rms':float(a.square().mean().sqrt()),
         'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))}

config=Qwen3VLConfig.from_pretrained(str(MODEL)).text_config;config.num_hidden_layers=50;config.use_cache=False;config._attn_implementation='sdpa'
with torch.device('meta'):model=Qwen3VLTextModel(config)
model.embed_tokens=nn.Identity();model.rotary_emb=Qwen3VLTextRotaryEmbedding(config)
wanted=set(dict(model.named_parameters()));state={};index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
for shard in sorted(set(index.values())):
 keys=[k for k,s in index.items() if s==shard and k.startswith('model.language_model.') and k[len('model.language_model.'):] in wanted]
 if not keys:continue
 with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
  for k in keys:state[k[len('model.language_model.'):]]=f.get_tensor(k).bfloat16()
missing,unexpected=model.load_state_dict(state,strict=False,assign=True);assert not missing and not unexpected,(missing,unexpected)
model.eval();inp=torch.load(CAP/'text_inputs.pt',weights_only=True);native_rope=torch.load(CAP/'text_rotary.pt',weights_only=True)
mask=inp['visual_pos_masks'];late_input={};active_mode='';report=None;outpath=None
norm_methods={};rope_method=model.rotary_emb.forward

def fp32_linear(self,x):return torch.nn.functional.linear(x.float(),self.weight.float(),self.bias.float() if self.bias is not None else None).to(x.dtype)
def native_norm(self,x):
 xf=x.float();out=xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+self.variance_epsilon)
 return (out*self.weight.float()).to(x.dtype)
def native_rotary(self,x,position_ids):return tuple(v.to(x.device,x.dtype) for v in native_rope)
for module in model.modules():
 if isinstance(module,nn.Linear):module.forward=types.MethodType(fp32_linear,module)
 if isinstance(module,Qwen3VLTextRMSNorm):norm_methods[module]=module.forward

def pre(index):
 def capture(module,args,kwargs):
  if index==49:
   payload={'hidden_states':args[0].detach().clone(),'position_embeddings':tuple(v.detach().clone() for v in kwargs['position_embeddings'])}
   if kwargs.get('attention_mask') is not None:payload['attention_mask']=kwargs['attention_mask'].detach().clone()
   late_input[index]=payload
 return capture

def post(index):
 def compare(module,args,value):
  item={'layer':index,'output_dtype':str(value.dtype),'output_rms':float(value.float().square().mean().sqrt()),'output_absmax':float(value.abs().max())}
  path=CAP/f'text_layer_{index:02d}.pt'
  if path.exists():
   actual=torch.load(path,weights_only=True);item['vs_actual_native']=metric(actual,value)
   item['text_positions']=metric(actual[~mask],value[~mask]);item['visual_positions']=metric(actual[mask],value[mask])
  if index>=24 or path.exists():report['layers'].append(item)
  if index==49:
   payload=late_input.pop(index);payload['output']=value.detach().clone()
   target=ROOT/f'cpu_last_{active_mode}.pt'
   if target.exists():raise RuntimeError('Refuse late-layer tensor overwrite')
   byte_count=sum(v.numel()*v.element_size() for v in (payload['hidden_states'],payload['output'],*payload['position_embeddings']))
   if 'attention_mask' in payload:byte_count+=payload['attention_mask'].numel()*payload['attention_mask'].element_size()
   assert byte_count<512*1024*1024
   with target.open('xb') as f:torch.save(payload,f)
   report['last_layer_capture']={'file':target.name,'tensor_bytes':byte_count,'file_bytes':target.stat().st_size}
  outpath.write_text(json.dumps(report,indent=2)+'\n')
  if path.exists():print(json.dumps({'mode':active_mode,**item}),flush=True)
 return compare
for i,layer in enumerate(model.layers):
 layer.register_forward_pre_hook(pre(i),with_kwargs=True);layer.register_forward_hook(post(i))
for mode in modes:
 active_mode=mode;outpath=ROOT/f'cpu_ab_{mode}.json'
 report={'mode':mode,'layers':[],'scope':'CPU FP32-linear oracle, BF16 parameters/I-O and shared round54 visual inputs; norm/RoPE ablation, not unchanged BF16 HF or video-quality proof'}
 for module,method in norm_methods.items():module.forward=types.MethodType(native_norm,module) if mode in ('norm-native','both-native') else method
 model.rotary_emb.forward=types.MethodType(native_rotary,model.rotary_emb) if mode in ('rope-native','both-native') else rope_method
 try:
  start=time.perf_counter()
  with torch.inference_mode():model(inputs_embeds=inp['inputs_embeds'],position_ids=inp['positions'],visual_pos_masks=mask,deepstack_visual_embeds=inp['deepstack_visual_embeds'],use_cache=False)
  report['forward_s']=time.perf_counter()-start;report['completed']=True
 except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
 finally:outpath.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',outpath,flush=True)
