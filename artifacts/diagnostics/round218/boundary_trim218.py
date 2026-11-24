#!/usr/bin/env python3
"""Boundary checks for one all-invalid key tile and a partially-valid key tile.

Uses already-pinned sources and independent kernels. Tensor outputs stay on ts-217;
reports are scalar. These tests run only after the model measurement has finished.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch_npu
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['base','trim'],required=True);args=p.parse_args()
root=Path('/tmp/h3_trim218')
label='boundary_'+args.variant
out=root/label;out.mkdir(mode=0o700,exist_ok=False)
os.environ['H3_FA215_KERNEL_DIR']=str(root/'kernels'/label)
os.environ['H3_FA215_PKG']=str(root/'pkg')
os.environ['H3_FA_BR_CAP']='128';os.environ['H3_FA_BC_CAP']='304'
if args.variant=='trim':os.environ['H3_FA_TRIM_Q_LIMIT']='2048'
else:os.environ.pop('H3_FA_TRIM_Q_LIMIT',None)
import fa218trim as fa
torch.npu.set_device(0)
generator=torch.Generator(device='cpu').manual_seed(20261124)
inputs=[torch.randn((1,7,2048,128),dtype=torch.float16,generator=generator).to('npu:0') for _ in range(3)]
records=[]
for pad in (256,257):
    values=fa.flash_attention_910a(*inputs,n_pad=pad)
    torch.npu.synchronize()
    host=values.detach().cpu().contiguous()
    words=host.view(torch.int16).numpy().view(np.uint16)
    row={'pad':pad,'valid_keys':2048-pad,'elements':int(host.numel()),
         'output_sha256':hashlib.sha256(words.tobytes()).hexdigest(),
         'nonfinite':int(((words&0x7fff)>=0x7c00).sum()),
         'saturated':int(((words&0x7fff)==0x7bff).sum())}
    if args.variant=='base':torch.save(host,out/f'pad{pad}_host_only.pt')
    else:
        base=torch.load(root/'boundary_base'/f'pad{pad}_host_only.pt',map_location='cpu',weights_only=True).contiguous()
        different=int((base.view(torch.int16)!=host.view(torch.int16)).sum())
        row['differing_words']=different
    records.append(row)
report={'variant':args.variant,'checks':records,'candidate_adopted':False}
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
raise SystemExit(0 if all(r.get('differing_words',0)==0 and r['nonfinite']==r['saturated']==0 for r in records) else 2)
