"""Experimental causal TBE wrapper, separate from production noncausal FA."""
import torch
import candidate_causal_fa910a as fa
from h3_text_causal57 import supported

def causal_tbe(q,k,v):
 if not supported(q,k,v) or q.dtype not in (torch.float16,torch.bfloat16):
  raise ValueError('Unsupported shape/dtype for experimental causal TBE')
 n=q.shape[-2];padded=((n+127)//128)*128
 with torch.autocast(device_type=q.device.type,enabled=False):
  qf=q.half();kf=k.half();vf=v.half()
  groups=q.shape[1]//k.shape[1]
  if groups!=1:kf=kf.repeat_interleave(groups,dim=1);vf=vf.repeat_interleave(groups,dim=1)
  if padded!=n:qf,kf,vf=fa._pad_qkv(qf,kf,vf,padded)
  out=fa._flash_attention_core(qf,kf,vf,key_seq_limit=n,causal=True)
  return out[:,:,:n].contiguous().to(q.dtype)
