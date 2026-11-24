#!/usr/bin/env python3
"""Compare host-only local FA outputs on ts-217; emit scalar statistics only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
root=Path('/tmp/h3_fa217/out')
base=torch.load(root/'small_base'/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
new=torch.load(root/'small_pv'/'output_host_only.pt',map_location='cpu',weights_only=True).contiguous()
assert base.shape==new.shape and base.dtype==new.dtype==torch.float16
aw=base.view(torch.int16).numpy().astype(np.uint16)
bw=new.view(torch.int16).numpy().astype(np.uint16)
a=base.float().numpy().astype(np.float64)
b=new.float().numpy().astype(np.float64)
delta=b-a
stats={'shape':list(base.shape),'dtype':str(base.dtype),'elements':base.numel(),
       'differing_words':int((aw!=bw).sum()),'differing_values':int((a!=b).sum()),
       'differing_fraction':float((aw!=bw).mean()),
       'relative_rms':float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(a.ravel()),1e-30)),
       'max_abs_difference':float(np.abs(delta).max()),
       'baseline_nonfinite_words':int(((aw&0x7fff)>=0x7c00).sum()),
       'candidate_nonfinite_words':int(((bw&0x7fff)>=0x7c00).sum()),
       'baseline_saturated_words':int(((aw&0x7fff)==0x7bff).sum()),
       'candidate_saturated_words':int(((bw&0x7fff)==0x7bff).sum()),
       'candidate_enabled':False,'disposition':'Awaiting user numeric decision; no promotion',
       'scope':'small synthetic local operator only, not model output'}
(root.parent/'comparison_pv217.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats))
