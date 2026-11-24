#!/usr/bin/env python3
"""Finite edge/mask tests for padding trim; no tensor moves off ts-217.

Nonfinite input injection is intentionally excluded: original strict QKV checks
would fail before FA. Finite invalid K/V are permitted and remain masked exactly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu

p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','trim'],required=True)
args=p.parse_args()
root=Path('/tmp/h3_gate219')
label='edge_'+args.variant
out=root/label;out.mkdir(mode=0o700,exist_ok=False)
os.environ['H3_FA215_KERNEL_DIR']=str(root/'edge_kernels'/label)
os.environ['H3_FA215_PKG']=str(root/'pkg')
os.environ['H3_FA_BR_CAP']='128';os.environ['H3_FA_BC_CAP']='304'
if args.variant=='trim':os.environ['H3_FA_TRIM_Q_LIMIT']='2048'
else:os.environ.pop('H3_FA_TRIM_Q_LIMIT',None)
import fa218trim as fa
torch.npu.set_device(0)
shape=(1,7,2048,128)
generator=torch.Generator(device='cpu').manual_seed(20261124)
raw=[torch.randn(shape,dtype=torch.float16,generator=generator) for _ in range(3)]
records=[]
# These values stay finite while probing signed-zero, exponent underflow, score
# finite-min and excluded-tail V behavior. Same FP16 words are used in both processes.
for name in ('random_tail_max','zero_signed_v','subnormal_v','skewed_negative_scores'):
    host=[x.clone() for x in raw]
    if name=='zero_signed_v':
        host[0].zero_();host[1].zero_()
        words=np.where(np.indices(shape)[-1]%2==0,0x8000,0).astype(np.uint16)
        host[2]=torch.from_numpy(words.view(np.float16).copy())
    elif name=='subnormal_v':
        words=np.where(np.indices(shape)[-1]%2==0,0x0001,0x8001).astype(np.uint16)
        host[2]=torch.from_numpy(words.view(np.float16).copy())
    elif name=='skewed_negative_scores':
        host[0].fill_(32.0);host[1].fill_(-32.0)
        host[2]=torch.where(torch.arange(shape[-1])%2==0,
                            torch.tensor(1.0,dtype=torch.float16),torch.tensor(-1.0,dtype=torch.float16)).expand(shape).clone()
    for pad in (256,257):
        valid=2048-pad
        tensors=[x.clone() for x in host]
        # Finite maximal invalid KV does not feed actual probabilities. QK may
        # overflow inside invalid columns, but the original mask is applied before
        # rowmax/softmax; candidate must preserve retained partial-mask behavior.
        if name=='random_tail_max':
            tensors[1][:,:,valid:,:].fill_(65504.0)
            tensors[2][:,:,valid:,:].fill_(-65504.0)
        inputs=[x.to('npu:0') for x in tensors]
        result=fa.flash_attention_910a(*inputs,n_pad=pad)
        torch.npu.synchronize()
        output=result.detach().cpu().contiguous()
        words=output.view(torch.int16).numpy().view(np.uint16)
        magnitude=words&0x7fff
        row={'name':name,'pad':pad,'valid_keys':valid,'elements':output.numel(),
             'output_sha256':hashlib.sha256(words.tobytes()).hexdigest(),
             'nonfinite_words':int((magnitude>=0x7c00).sum()),
             'saturated_words':int((magnitude==0x7bff).sum()),
             'negative_zero_words':int((words==0x8000).sum())}
        filename=f'{name}_pad{pad}_host_only.pt'
        if args.variant=='base':torch.save(output,out/filename)
        else:
            reference=torch.load(root/'edge_base'/filename,map_location='cpu',weights_only=True).contiguous()
            differing=int((reference.view(torch.int16)!=output.view(torch.int16)).sum())
            row['differing_words']=differing
        records.append(row)
        (out/'result.json').write_text(json.dumps({'variant':args.variant,'checks':records,'all_candidate_results_unused':True},indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if row['nonfinite_words'] or row.get('differing_words',0):
            print('STOP: failed finite/word gate, candidate stays disabled',flush=True)
            raise SystemExit(2)
print('all edge cases complete')
