#!/usr/bin/env python3
"""Actual pre-RoPE inputs: native fused vs HF two-product rounds vs custom TBE."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from vllm_omni.platforms.npu.layers.rotary_embedding import npu_rotary_mul_with_bsnd_fallback
from rope_vector60 import rope

ROOT = Path('/tmp/h3_round60_rope01')
OUT = ROOT/'rope_validation02.json'
if OUT.exists():
    raise RuntimeError('Refuse RoPE evidence overwrite')
torch.npu.set_device(0)
torch.set_num_threads(16)
r = {'cases': [], 'boundary_cases': [], 'scope': 'Actual stable text pre-RoPE Q/K with fixed tables; per-operator contracts, not generation-lossless proof'}


def rotate(x):
    return torch.cat((-x[..., 64:], x[..., :64]), dim=-1)


def hf_eager(x, c, s):
    with torch.autocast(device_type=x.device.type, enabled=False):
        a = (x.float()*c.float().unsqueeze(1)).to(x.dtype)
        b = (rotate(x.float())*s.float().unsqueeze(1)).to(x.dtype)
        return (a.float()+b.float()).to(x.dtype)


def once_eager(x, c, s):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return (x.float()*c.float().unsqueeze(1)+rotate(x.float())*s.float().unsqueeze(1)).to(x.dtype)


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    e = x.float().double()-y.float().double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'finite': bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
            'max_abs_error': float(e.abs().max()), 'relative_rms': float(e.square().mean().sqrt()/y.float().double().square().mean().sqrt().clamp_min(1e-30)),
            'word_mismatches': int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'zero_sign_mismatches': int((x.float().eq(0)&y.float().eq(0)&torch.signbit(x.float()).ne(torch.signbit(y.float()))).sum()),
            'elements': x.numel()}


def timed(fn):
    fn();torch.npu.synchronize();times=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(10):out=fn()
        torch.npu.synchronize();times.append((time.perf_counter()-t)*1000/10)
    return out, {'groups_ms':times,'median_ms':statistics.median(times)}


try:
    with torch.inference_mode():
        for layer in (0,24,49):
            d=torch.load(ROOT/'capture'/f'rope_layer{layer:02d}.pt',weights_only=True)
            c,s=d['cos'].to('npu'),d['sin'].to('npu')
            for kind in ('q','k'):
                xc=d[kind];x=xc.transpose(1,2).contiguous().to('npu').transpose(1,2)
                hf=xc*d['cos'].unsqueeze(1)+rotate(xc)*d['sin'].unsqueeze(1)
                once=(xc.float()*d['cos'].float().unsqueeze(1)+rotate(xc.float())*d['sin'].float().unsqueeze(1)).bfloat16()
                row={'layer':layer,'kind':kind,'shape':list(x.shape),'input_stride':list(x.stride()),
                     'captured_native_vs_cpu_hf':metric(d[kind+'_output'],hf),
                     'captured_native_vs_cpu_once':metric(d[kind+'_output'],once),'variants':{}}
                for label,fn in [('native_npu',lambda:npu_rotary_mul_with_bsnd_fallback(x,c,s,1)),
                                 ('hf_eager',lambda:hf_eager(x,c,s)),('fp32_once_eager',lambda:once_eager(x,c,s)),
                                 ('tbe_native',lambda:rope(x,c,s,False)),('tbe_hf',lambda:rope(x,c,s,True))]:
                    try:
                        out,times=timed(fn)
                        row['variants'][label]={'timing':times,'vs_cpu_hf':metric(out,hf),'vs_cpu_once':metric(out,once),
                                                'vs_actual_native':metric(out,d[kind+'_output'])}
                    except Exception as exc:
                        row['variants'][label]={'failure':{'type':type(exc).__name__,'message':str(exc)}}
                r['cases'].append(row);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(row),flush=True)
        torch.manual_seed(60)
        for n in (1,7,129):
            for h in (1,8):
                x=torch.randn(1,h,n,128).bfloat16().to('npu')
                c=torch.randn(1,n,128).bfloat16().to('npu');s=torch.randn_like(c)
                native=npu_rotary_mul_with_bsnd_fallback(x,c,s,1)
                hf=hf_eager(x,c,s)
                a=rope(x,c,s,False);b=rope(x,c,s,True)
                torch.npu.synchronize()
                row={'n':n,'h':h,'tbe_native_vs_native':metric(a,native),'tbe_hf_vs_hf_eager':metric(b,hf)}
                r['boundary_cases'].append(row)
        r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
