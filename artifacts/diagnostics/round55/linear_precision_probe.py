#!/usr/bin/env python3
"""Same actual vision inputs/weights: isolate linear accumulation/bias rounding.

No production edits or precision flags are changed. Compare both CPU BF16 HF
reference and CPU FP32 computation rounded to BF16. NPU weights remain sourced
from the exact BF16 checkpoint; each path's casts are part of timed execution.
"""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

ROOT=Path('/tmp/h3_round55_vision01');MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder');OUT=ROOT/'linear_precision_probe.json'
if OUT.exists():raise RuntimeError('Refuse existing linear precision evidence')
torch.npu.set_device(0);torch.set_num_threads(16)
report={'operators':[],'npu_matmul_allow_hf32':torch.npu.matmul.allow_hf32,
        'scope':'Matched actual block0 vision tensor and weight comparisons. Timing includes conversion; individual linear probes are not full-model quality/performance proof.'}

def metric(x,y):
 x=x.float().cpu();y=y.float().cpu();e=x-y;finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
 return {'finite':finite,'bit_exact':bool(torch.equal(x,y)),'max_abs_error':float(e.abs().max()) if finite else None,
         'relative_rms':float(e.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12)) if finite else None}
try:
 config=Qwen3VLConfig.from_pretrained(str(MODEL)).vision_config;config._attn_implementation='sdpa'
 teacher=Qwen3VLVisionBlock(config).to(torch.bfloat16).eval();index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map'];prefix='model.visual.blocks.0.';state={}
 for shard in sorted({s for k,s in index.items() if k.startswith(prefix)}):
  with safe_open(str(MODEL/shard),framework='pt',device='cpu') as f:
   for k,s in index.items():
    if s==shard and k.startswith(prefix):state[k[len(prefix):]]=f.get_tensor(k).to(torch.bfloat16)
 teacher.load_state_dict(state,strict=True);data=torch.load(ROOT/'block_00.pt',weights_only=True)
 with torch.inference_mode():
  norm1=teacher.norm1(data['hidden']);attention=teacher.attn(norm1,cu_seqlens=data['cu_seqlens'],position_embeddings=data['position_embeddings'])
  norm2=teacher.norm2(data['hidden']+attention);fc1=teacher.mlp.linear_fc1(norm2);gelu=teacher.mlp.act_fn(fc1)
  jobs=[('qkv',norm1,teacher.attn.qkv),('mlp_fc1',norm2,teacher.mlp.linear_fc1),('mlp_fc2',gelu,teacher.mlp.linear_fc2)]
  for name,x,layer in jobs:
   bf16_reference=F.linear(x,layer.weight,layer.bias)
   fp32_reference=F.linear(x.float(),layer.weight.float(),layer.bias.float()).bfloat16()
   xn=x.to('npu');wn=layer.weight.to('npu');bn=layer.bias.to('npu')
   variants={
    'native_bf16':lambda:F.linear(xn,wn,bn),
    'cast_fp16_linear_then_bf16':lambda:F.linear(xn.half(),wn.half(),bn.half()).bfloat16(),
    'explicit_fp32_linear_then_bf16':lambda:F.linear(xn.float(),wn.float(),bn.float()).bfloat16(),
    'fp32_dot_then_bias_then_bf16':lambda:(F.linear(xn.float(),wn.float(),None)+bn.float()).bfloat16(),
    'native_bf16_dot_then_fp32_bias':lambda:(F.linear(xn,wn,None).float()+bn.float()).bfloat16()}
   item={'name':name,'input_shape':list(x.shape),'weight_shape':list(layer.weight.shape),
         'cpu_bf16_vs_cpu_fp32_rounded':metric(bf16_reference,fp32_reference),'variants':{}}
   for label,call in variants.items():
    call();torch.npu.synchronize();times=[]
    for _ in range(3):
     t=time.perf_counter();out=call();torch.npu.synchronize();times.append((time.perf_counter()-t)*1000)
    cpu=out.cpu();item['variants'][label]={'to_hf_bf16':metric(cpu,bf16_reference),'to_fp32_rounded':metric(cpu,fp32_reference),
                                        'groups_ms':times,'median_ms':statistics.median(times)}
   report['operators'].append(item);OUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(item),flush=True)
   del xn,wn,bn,out
 report['completed']=True
except BaseException as exc:
 report['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
 OUT.write_text(json.dumps(report,indent=2)+'\n');print('Evidence:',OUT,flush=True)
