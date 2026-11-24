#!/usr/bin/env python3
"""Report direct word differences on the host only; no output archive transfer."""
import argparse
import json
import numpy as np
from pathlib import Path
import torch
parser=argparse.ArgumentParser()
parser.add_argument('--baseline',required=True)
parser.add_argument('--candidate',required=True)
args=parser.parse_args()
root=Path('/tmp/h3_broadcast218/out')
a=torch.load(root/args.baseline/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
b=torch.load(root/args.candidate/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
assert a.shape==b.shape and a.dtype==b.dtype==torch.float16
wa=a.view(torch.int16).numpy().reshape(-1).view(np.uint16)
wb=b.view(torch.int16).numpy().reshape(-1).view(np.uint16)
different=0;bad=[0,0];sat=[0,0]
for start in range(0,wa.size,1000000):
 aa=wa[start:start+1000000];bb=wb[start:start+1000000]
 different+=int((aa!=bb).sum())
 for i,w in enumerate([aa,bb]):
  magnitude=w&0x7fff
  bad[i]+=int((magnitude>=0x7c00).sum());sat[i]+=int((magnitude==0x7bff).sum())
record={'baseline':args.baseline,'candidate':args.candidate,'elements':int(a.numel()),'differing_words':different,
        'baseline_nonfinite':bad[0],'candidate_nonfinite':bad[1],
        'baseline_saturated':sat[0],'candidate_saturated':sat[1],
        'automatic_adoption':False,'host_only_tensors':True}
out=root.parent/f'comparison_{args.baseline}_{args.candidate}.json'
assert not out.exists()
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(0 if different==0 and not any(bad+sat) else 2)
