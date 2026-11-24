#!/usr/bin/env python3
"""Numerical scatter is reported, never used to promote an unproven candidate."""
import json
from pathlib import Path
import torch
root=Path('/tmp/h3_path218/path218')
pairs=[('prediction_case0_step25_repeat0.pt','prediction_case0_step25_repeat1.pt','native_self_repeat'),
       ('prediction_case0_step25_repeat0.pt','prediction_case1_step25_repeat0.pt','instrumentation_vs_clean'),
       ('prediction_case0_step25_repeat0.pt','prediction_case2_step25_repeat1.pt','restored_vs_clean')]
rows=[]
for a_path,b_path,label in pairs:
 a=torch.load(root/a_path,map_location='cpu',weights_only=True)
 b=torch.load(root/b_path,map_location='cpu',weights_only=True)
 metrics={}
 for key in ('video_velocity','audio_velocity'):
  aa=a[key].contiguous();bb=b[key].contiguous()
  x=aa.double();y=bb.double();delta=y-x
  metrics[key]={'different_words':int((aa.view(torch.int32)!=bb.view(torch.int32)).sum()),
                'elements':aa.numel(),'relative_rms':float(delta.norm()/x.norm().clamp_min(1e-30)),
                'max_abs':float(delta.abs().max()),'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all())}
 rows.append({'label':label,'metrics':metrics,'not_a_numeric_tolerance':True})
p=root.parent/'selfrepeat_metrics218.json';assert not p.exists();p.write_text(json.dumps({'comparisons':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
