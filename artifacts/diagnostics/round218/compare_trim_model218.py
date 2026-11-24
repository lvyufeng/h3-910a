#!/usr/bin/env python3
"""Report same-run final-prediction scatter; exact FA leaf gate is a separate result."""
import json
from pathlib import Path
import torch
root=Path('/tmp/h3_trim_model218/single_trim218')
base=torch.load(root/'prediction_case0_step25_repeat0.pt',map_location='cpu',weights_only=True)
comparisons=[]
for case,repeat,label in [(0,1,'native_self_repeat'),(1,0,'trim_vs_native_first'),
                          (1,1,'trim_vs_native_second'),(2,1,'restored_vs_native')]:
    other=torch.load(root/f'prediction_case{case}_step25_repeat{repeat}.pt',map_location='cpu',weights_only=True)
    metrics={}
    for key in ('video_velocity','audio_velocity'):
        a=base[key].contiguous();b=other[key].contiguous()
        x=a.double();y=b.double();delta=y-x
        metrics[key]={'different_words':int((a.view(torch.int32)!=b.view(torch.int32)).sum()),
                      'elements':a.numel(),'relative_rms':float(delta.norm()/x.norm().clamp_min(1e-30)),
                      'max_abs_difference':float(delta.abs().max()),
                      'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all())}
    comparisons.append({'label':label,'metrics':metrics})
report={'comparisons':comparisons,'native_repeat_is_not_a_tolerance':True,
        'no_full_model_numeric_promotion':True,'exact_FA_real_input_gate_reported_separately':True}
path=root.parent/'trim_prediction_metrics218.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
