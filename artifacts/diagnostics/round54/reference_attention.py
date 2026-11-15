#!/usr/bin/env python3
"""Full-key FP32 reference checks on captured real text/DiT attention tensors."""
import json
from pathlib import Path
import time
import torch

ROOT=Path('/tmp/h3_round54_capture01')
OUT=ROOT/'attention_reference.json'
if OUT.exists():raise RuntimeError('Refuse to overwrite a reference result')
torch.set_num_threads(16)
report={'cases':[],
        'text_capture_status':'Not captured: runtime replaces base SDPA with the NPU GQA function; do not treat the base-source hook as effective-path evidence',
        'scope':'Sampled real query rows, all valid keys and all captured rank-zero heads; not full-model quality proof'}

def stats(x):
    dtype=str(x.dtype); x=x.float(); finite=torch.isfinite(x)
    return {'shape':list(x.shape),'dtype':dtype,'finite':bool(finite.all()),
            'nonfinite_count':int((~finite).sum()),'max_abs':float(x.abs().max()) if bool(finite.all()) else None,
            'rms':float(x.square().mean().sqrt()) if bool(finite.all()) else None}

def error(actual,expected):
    if not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
        return {'finite':False,'max_abs_error':None,'relative_rms':None,'reference_rms':None}
    diff=actual.float()-expected.float()
    return {'finite':True,'max_abs_error':float(diff.abs().max()),
            'relative_rms':float(diff.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12)),
            'reference_rms':float(expected.float().square().mean().sqrt())}

try:
    for name in ('dit_attention0','dit_attention49'):
        if not (ROOT/(name+'.json')).exists():raise RuntimeError('Capture is not complete: '+name)
        data=torch.load(ROOT/(name+'.pt'),weights_only=True)
        q,k,v,o=[data[x] for x in ('query','key','value','output')]
        original_dtype=str(q.dtype)
        b,h,n,d=q.shape; valid=k.shape[2]-data['n_pad']
        assert b==1 and valid>0
        if k.shape[1]!=h:
            k=k.repeat_interleave(h//k.shape[1],dim=1);v=v.repeat_interleave(h//v.shape[1],dim=1)
        indices=sorted({min(valid-1,i) for i in (0,1,2,7,31,valid//2,valid-2,valid-1)})
        query_indices=torch.tensor(indices)
        actual=o[:,:,query_indices,:].float()
        case={'name':name,'matching_call_ordinal':data['matching_call_ordinal'],'source_dtype':original_dtype,
              'causal':data['causal'],'valid_keys':valid,'original_pad':data['n_pad'],'query_indices':indices,
              'input_stats':{key:stats(value) for key,value in zip(('query','key','value','output'),(q,k,v,o))},
              'padding_key_max_abs':float(k[:,:,valid:,:].float().abs().max()) if valid<k.shape[2] else None,
              'padding_value_max_abs':float(v[:,:,valid:,:].float().abs().max()) if valid<v.shape[2] else None,
              'references':{}}
        for mode in ('fp32_exact_scale','fp16_q_rounding') if not data['causal'] else ('fp32_exact_scale',):
            qq=q[:,:,query_indices,:].float()*data['scale']
            if mode=='fp16_q_rounding':qq=qq.half().float()
            start=time.perf_counter()
            logits=qq@k[:,:,:valid,:].float().transpose(-2,-1)
            if data['causal']:
                mask=torch.arange(valid)[None,:]>query_indices[:,None]
                logits=logits.masked_fill(mask[None,None,:,:],float('-inf'))
            probabilities=torch.softmax(logits,dim=-1)
            expected=probabilities@v[:,:,:valid,:].float()
            finite_scores=logits[torch.isfinite(logits)]
            case['references'][mode]={**error(actual,expected),
               'sampled_score_min':float(finite_scores.min()),'sampled_score_max':float(finite_scores.max()),
               'max_probability':float(probabilities.max()),'reference_wall_s':time.perf_counter()-start}
        report['cases'].append(case)
        print(json.dumps(case),flush=True)
        del q,k,v,o,data,actual,logits,expected,probabilities
    report['completed']=True
except BaseException as exc:
    report['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Evidence:',OUT,flush=True)
