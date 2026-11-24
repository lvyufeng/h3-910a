#!/usr/bin/env python3
"""Chunked exact-word checks of corrected PV variants; outputs never leave ts-217."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

root=Path('/tmp/h3_fa217/out')
pairs=[('small_base','small_pvgroup'),('small_base','small_pvdouble'),
       ('real_restored_final','real_pvgroup'),('real_restored_final','real_pvdouble'),
       ('real_restored_final','real_restored_end')]
records=[]
for base_label,new_label in pairs:
    base=torch.load(root/base_label/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
    new=torch.load(root/new_label/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
    assert base.shape==new.shape and base.dtype==new.dtype==torch.float16
    a=base.view(torch.int16).numpy().reshape(-1).view(np.uint16)
    b=new.view(torch.int16).numpy().reshape(-1).view(np.uint16)
    differing=0;nonfinite=[0,0];saturation=[0,0]
    for start in range(0,a.size,1_000_000):
        aa=a[start:start+1_000_000];bb=b[start:start+1_000_000]
        differing+=int((aa!=bb).sum())
        for i,words in enumerate([aa,bb]):
            magnitude=words&0x7fff
            nonfinite[i]+=int((magnitude>=0x7c00).sum())
            saturation[i]+=int((magnitude==0x7bff).sum())
    records.append({'baseline':base_label,'candidate':new_label,'elements':int(a.size),
                    'differing_words':differing,'baseline_nonfinite':nonfinite[0],
                    'candidate_nonfinite':nonfinite[1],'baseline_saturated':saturation[0],
                    'candidate_saturated':saturation[1],
                    'base_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                    'candidate_sha256':hashlib.sha256(b.tobytes()).hexdigest()})
    print(json.dumps(records[-1]),flush=True)
    del base,new,a,b
path=root.parent/'exact_comparison217.json'
assert not path.exists()
path.write_text(json.dumps({'comparisons':records,'no_model_or_client_tensor_transfer':True},indent=2)+'\n')
