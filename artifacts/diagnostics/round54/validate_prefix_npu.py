#!/usr/bin/env python3
"""Validate actual 910A sub-column masking, real QKV accuracy and wrapper timing."""
import importlib.util
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu

ROOT=Path('/tmp/h3_round54_validation01');CAP=Path('/tmp/h3_round54_capture01')
OUT=ROOT/'prefix_validation.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite validation evidence')
def load(file,name):
    s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
new=load(ROOT/'candidate_prefix_fa910a.py','prefix54')
old=load(ROOT/'source/fa910a.py','old54')
torch.npu.set_device(0);torch.manual_seed(54)
r={'tail_tests':[],'real_cases':[],'timings':[],'scope':'Operator prefix correctness and same-input wrapper A/B; not generation-quality proof'}

def ref(q,k,v,scale,valid,indices):
    qq=(q.cpu()[:,:,indices,:].float()*scale).half().float()
    return torch.softmax(qq@k.cpu()[:,:,:valid,:].float().transpose(-2,-1),dim=-1)@v.cpu()[:,:,:valid,:].float()
def err(x,y):
    x=x.float();y=y.float();d=x-y
    return {'finite':bool(torch.isfinite(x).all()),'max_abs_error':float(d.abs().max()),
            'relative_rms':float(d.square().mean().sqrt()/y.square().mean().sqrt().clamp_min(1e-12))}

try:
    for valid in (1,15,17,303,305,2417,2431):
        n=2432
        q=torch.full((1,1,n,128),2.,dtype=torch.float16,device='npu')
        k=torch.full_like(q,3.);v=torch.full_like(q,7.)
        # Whole contiguous CPU initialization avoids the driver's offset fill warning.
        kc=k.cpu();vc=v.cpu();kc[:,:,:valid,:].fill_(-2.);vc[:,:,:valid,:].fill_(1.)
        k=kc.to('npu');v=vc.to('npu')
        out=new.flash_attention_910a(q,k,v,n_pad=n-valid)
        torch.npu.synchronize();cpu=out.cpu()
        item={'valid_keys':valid,'original_pad':n-valid,'max_error_to_one':float((cpu.float()-1.).abs().max()),
              'finite':bool(torch.isfinite(cpu).all())}
        r['tail_tests'].append(item);print(json.dumps(item),flush=True)
        assert item['finite'] and item['max_error_to_one']<.002
        # A last-valid-key impulse ensures the boundary mask did not simply
        # discard valid columns while removing the nonzero invalid tail.
        vc.zero_();vc[:,:,valid-1,:]=float(valid);vc[:,:,valid:,:].fill_(7.)
        v=vc.to('npu')
        impulse=new.flash_attention_910a(q,k,v,n_pad=n-valid)
        torch.npu.synchronize();expected_impulse=float(torch.tensor(valid,dtype=torch.float16).float()/valid)
        impulse_error=float((impulse.cpu().float()-expected_impulse).abs().max())
        assert impulse_error<.002
        item['last_valid_key_impulse_error']=impulse_error
    for name in ('dit_attention0','dit_attention49'):
        data=torch.load(CAP/(name+'.pt'),weights_only=True)
        q,k,v=[data[x].to('npu') for x in ('query','key','value')]
        valid=q.shape[2]-data['n_pad'];indices=[0,1,2,7,31,valid//2,valid-2,valid-1]
        outputs={}
        for label,fa in (('old_zero_key_lse',old),('new_exact_prefix',new)):
            for _ in range(2):out=fa.flash_attention_910a(q,k,v,scale=data['scale'],n_pad=data['n_pad'])
            torch.npu.synchronize();outputs[label]=out.clone().cpu()
            if name=='dit_attention49':
                groups=[]
                for _ in range(3):
                    t=time.perf_counter()
                    for __ in range(3):out=fa.flash_attention_910a(q,k,v,scale=data['scale'],n_pad=data['n_pad'])
                    torch.npu.synchronize();groups.append((time.perf_counter()-t)*1000/3)
                r['timings'].append({'name':name,'variant':label,'groups_ms':groups,'median_ms':statistics.median(groups)})
        expected=ref(q,k,v,data['scale'],valid,indices)
        case={'name':name,'valid_keys':valid,'query_indices':indices,
              'old':err(outputs['old_zero_key_lse'][:,:,indices,:],expected),
              'new':err(outputs['new_exact_prefix'][:,:,indices,:],expected)}
        r['real_cases'].append(case);print(json.dumps(case),flush=True)
        assert case['new']['finite']
        assert case['new']['relative_rms']<case['old']['relative_rms']
    r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
