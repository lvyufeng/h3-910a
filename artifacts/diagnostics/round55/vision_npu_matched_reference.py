#!/usr/bin/env python3
"""Teacher-forced NPU replay of real HF vision block inputs/weights.

Each isolated sub-operator consumes the matching CPU/HF input, rather than
accumulating upstream NPU differences. No production source is changed.
"""
import argparse
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
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

ROOT=Path('/tmp/h3_round55_vision01');CODE=ROOT/'source/encoder.py';MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
p=argparse.ArgumentParser();p.add_argument('--blocks',default='0,8,16,24,26');p.add_argument('--candidate',action='store_true');args=p.parse_args()
selected=tuple(map(int,args.blocks.split(',')));assert selected and all(x in (0,8,16,24,26) for x in selected)
if args.candidate:
 CODE=ROOT/'candidate_encoder.py';os.environ['VLLM_OMNI_910A_VISION_FP32']='1'
else:os.environ.pop('VLLM_OMNI_910A_VISION_FP32',None)
OUT=ROOT/(('npu_matched_candidate_' if args.candidate else 'npu_matched_reference_')+ '_'.join(map(str,selected))+'.json')
if OUT.exists():raise RuntimeError('Refuse existing matched-reference receipt')
torch.npu.set_device(0);torch.set_num_threads(16)
report={'blocks':[],'candidate':args.candidate,'scope':'Actual captured native CPU block inputs and installed HF CPU sub-operators; NPU sub-operators share exact HF inputs. Not full-model quality validation.'}

def metric(a,b):
 x=a.detach().float().cpu();y=b.detach().float().cpu();assert x.shape==y.shape;e=x-y
 finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
 return {'shape':list(x.shape),'finite':finite,'bit_exact':bool(torch.equal(x,y)),
         'max_abs_error':float(e.abs().max()) if finite else None,
         'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if finite else None}

def timed(call):
 torch.npu.synchronize();t=time.perf_counter();out=call();torch.npu.synchronize();return out,time.perf_counter()-t
try:
 tree=ast.parse(CODE.read_text());names={'_rotate_half','_apply_rotary_pos_emb_vision','_vision_sdpa_bf16_compatible','_NpuBF16CompatibleLayerNorm','_NpuBF16CompatibleLinear','_vision_fp32_enabled'}
 nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (n.name in names or n.name.startswith('MiniMaxH3Qwen3VLVision'))]
 ns={'torch':torch,'nn':nn,'F':F,'Any':Any};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(CODE),'exec'),ns)
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;config._attn_implementation='sdpa'
 weight_index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
 for index in selected:
  path=ROOT/f'block_{index:02d}.pt'
  if not path.exists():raise RuntimeError('CPU reference block not ready: '+str(index))
  data=torch.load(path,weights_only=True)
  prefix=f'model.visual.blocks.{index}.';state={}
  shards=sorted({file for key,file in weight_index.items() if key.startswith(prefix)})
  for shard in shards:
   with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
    for key,file in weight_index.items():
     if file==shard and key.startswith(prefix):state[key[len(prefix):]]=f.get_tensor(key).to(torch.bfloat16)
  teacher=Qwen3VLVisionBlock(config).to(torch.bfloat16).eval();teacher.load_state_dict(state,strict=True)
  native=ns['MiniMaxH3Qwen3VLVisionBlock'](config).to(torch.bfloat16).eval();native.load_state_dict(state,strict=True);native.to('npu')
  x=data['hidden'];cu=data['cu_seqlens'];pe=data['position_embeddings']
  xn=x.to('npu');cun=cu.to('npu');pen=tuple(t.to('npu') for t in pe)
  item={'block':index,'isolated':{}}
  with torch.inference_mode():
   norm1=teacher.norm1(x);norm1n,dt=timed(lambda:native.norm1(xn))
   item['isolated']['norm1']={**metric(norm1n,norm1),'npu_s':dt}
   qkv=teacher.attn.qkv(norm1);qkvn,dt=timed(lambda:native.attn.qkv(norm1.to('npu')))
   item['isolated']['qkv_linear_same_hf_norm_input']={**metric(qkvn,qkv),'npu_s':dt}
   # The native attention helper intentionally uses explicit FP32 math. Compare
   # both the native and an FP32 full-key CPU math on identical HF QKV.
   seq=x.shape[0];q,k,v=qkv.reshape(seq,3,config.num_heads,-1).permute(1,0,2,3).unbind(0)
   from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
   q,k=apply_rotary_pos_emb_vision(q,k,*pe)
   q,k,v=[t.transpose(0,1).unsqueeze(0) for t in (q,k,v)]
   length=(cu[1:]-cu[:-1]).tolist();split=[torch.split(t,length,dim=2) for t in (q,k,v)]
   math_outputs=[];cpu_sdpa_outputs=[];npu_math_outputs=[]
   for qc,kc,vc in zip(*split):
    cpu_sdpa_outputs.append(F.scaled_dot_product_attention(qc.contiguous(),kc.contiguous(),vc.contiguous(),dropout_p=0.,is_causal=False))
    math_outputs.append((torch.softmax(qc.float()@kc.float().transpose(-2,-1)/(qc.shape[-1]**.5),dim=-1)@vc.float()).to(qc.dtype))
    yn,dt=timed(lambda:ns['_vision_sdpa_bf16_compatible'](qc.to('npu'),kc.to('npu'),vc.to('npu')))
    npu_math_outputs.append(yn.cpu())
   ctorch=torch.cat(cpu_sdpa_outputs,dim=2);cmath=torch.cat(math_outputs,dim=2);nmath=torch.cat(npu_math_outputs,dim=2)
   item['isolated']['cpu_fp32_math_vs_cpu_sdpa_same_qkv']=metric(cmath,ctorch)
   item['isolated']['npu_fp32_math_vs_cpu_fp32_math_same_qkv']=metric(nmath,cmath)
   item['isolated']['npu_fp32_math_vs_cpu_sdpa_same_qkv']=metric(nmath,ctorch)
   attention=teacher.attn(norm1,cu_seqlens=cu,position_embeddings=pe)
   attentionn,dt=timed(lambda:native.attn(norm1.to('npu'),cu_seqlens=cun,position_embeddings=pen))
   item['isolated']['attention_same_hf_norm_input']={**metric(attentionn,attention),'npu_s':dt}
   residual=x+attention;norm2=teacher.norm2(residual)
   norm2n,dt=timed(lambda:native.norm2(residual.to('npu')))
   item['isolated']['norm2_same_hf_residual_input']={**metric(norm2n,norm2),'npu_s':dt}
   fc1=teacher.mlp.linear_fc1(norm2);fc1n,dt=timed(lambda:native.mlp.linear_fc1(norm2.to('npu')))
   item['isolated']['mlp_fc1_same_hf_norm_input']={**metric(fc1n,fc1),'npu_s':dt}
   gelu=teacher.mlp.act_fn(fc1);gelun,dt=timed(lambda:native.mlp.act_fn(fc1.to('npu')))
   item['isolated']['gelu_same_hf_fc1_input']={**metric(gelun,gelu),'npu_s':dt}
   fc2=teacher.mlp.linear_fc2(gelu);fc2n,dt=timed(lambda:native.mlp.linear_fc2(gelu.to('npu')))
   item['isolated']['mlp_fc2_same_hf_gelu_input']={**metric(fc2n,fc2),'npu_s':dt}
   whole,dt=timed(lambda:native(xn,cu_seqlens=cun,position_embeddings=pen))
   item['npu_whole_block_vs_hf_same_input']={**metric(whole,data['hf_cpu_output']),'npu_s':dt}
  report['blocks'].append(item);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
  del native,teacher,xn,data,q,k,v,cmath,nmath,ctorch,whole
  torch.npu.empty_cache()
 report['completed']=True
except BaseException as exc:
 report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',OUT,flush=True)
