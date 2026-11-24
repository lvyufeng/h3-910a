#!/usr/bin/env python3
"""Exact word gate for trimming variants; tensors stay on ts-217."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--candidate',required=True);args=p.parse_args()
r=Path('/tmp/h3_trim218/out')
a=torch.load(r/args.base/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
b=torch.load(r/args.candidate/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
assert a.shape==b.shape and a.dtype==b.dtype==torch.float16
wa=a.view(torch.int16).numpy().reshape(-1).view(np.uint16);wb=b.view(torch.int16).numpy().reshape(-1).view(np.uint16)
n=0;bad=[0,0];sat=[0,0]
for start in range(0,wa.size,1000000):
 aa=wa[start:start+1000000];bb=wb[start:start+1000000];n+=int((aa!=bb).sum())
 for i,w in enumerate([aa,bb]):
  magnitude=w&0x7fff;bad[i]+=int((magnitude>=0x7c00).sum());sat[i]+=int((magnitude==0x7bff).sum())
record={'base':args.base,'candidate':args.candidate,'elements':int(wa.size),'differing_words':n,
        'base_nonfinite':bad[0],'candidate_nonfinite':bad[1],'base_saturated':sat[0],'candidate_saturated':sat[1],
        'model_adoption':False,'no_tensor_transfer_to_client':True}
path=r.parent/f'comparison_{args.base}_{args.candidate}.json';assert not path.exists()
path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
raise SystemExit(0 if n==0 and not any(bad+sat) else 2)
