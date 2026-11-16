"""Opt-in FP32 causal GQA math for H3 text encoder on Ascend 910A.

Reference/comparison path for short self-attention. Dense scores are bounded by
sequence/query-head limits. No approximate sparsity, FP16 casts or tail trimming.
The default remains the existing NPU native GQA path.
"""
import os
import torch
from torch.nn import functional as F

_MASKS={}
_MAX_MASKS=4
_MAX_SEQUENCE=4096
_MAX_QUERY_HEADS=16

def _mask(n,device):
 key=(n,device)
 value=_MASKS.get(key)
 if value is None:
  if len(_MASKS)>=_MAX_MASKS:_MASKS.clear()
  value=torch.ones((n,n),device=device,dtype=torch.bool).triu(1)
  _MASKS[key]=value
 return value

def supported(q,k,v):
 return (q.ndim==k.ndim==v.ndim==4 and q.device.type in ('npu','privateuseone')
         and q.device==k.device==v.device and q.dtype==k.dtype==v.dtype
         and q.dtype in (torch.float16,torch.bfloat16,torch.float32)
         and q.shape[0]==k.shape[0]==v.shape[0]==1
         and q.shape[-2]==k.shape[-2]==v.shape[-2] and 0<q.shape[-2]<=_MAX_SEQUENCE
         and q.shape[-1]==k.shape[-1]==v.shape[-1] and q.shape[-1]==128
         and 0<k.shape[1]==v.shape[1]<=q.shape[1]<=_MAX_QUERY_HEADS
         and q.shape[1]%k.shape[1]==0)

def causal_fp32(q,k,v):
 if not supported(q,k,v):raise ValueError('Unsupported shape/dtype for bounded causal math')
 with torch.autocast(device_type=q.device.type,enabled=False):
  qf=q.float().contiguous();kf=k.float().contiguous();vf=v.float().contiguous()
  groups=q.shape[1]//k.shape[1]
  if groups!=1:
   kf=kf.repeat_interleave(groups,dim=1)
   vf=vf.repeat_interleave(groups,dim=1)
  scores=torch.matmul(qf,kf.transpose(-2,-1))*(q.shape[-1]**-0.5)
  scores=scores.masked_fill(_mask(q.shape[-2],q.device),-torch.inf)
  probabilities=torch.softmax(scores,dim=-1)
  return torch.matmul(probabilities,vf).to(q.dtype)

def attention(q,k,v):
 enabled=os.environ.get('VLLM_OMNI_910A_TEXT_CAUSAL_FP32','0').lower() in ('1','true','yes','on')
 if enabled and supported(q,k,v):return causal_fp32(q,k,v)
 return F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=True,enable_gqa=q.shape[1]!=k.shape[1])
