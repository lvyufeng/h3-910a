#!/usr/bin/env python3
"""Actual indexed FP16 AdaLN capture verification and complete-wrapper timing."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from vllm_omni.diffusion.layers import indexed_modulation as reference
from adaln_vector61 import launch

ROOT=Path('/tmp/h3_round61_adaln01')
OUT=ROOT/'adaln_validation.json'
if OUT.exists():raise RuntimeError('Refuse AdaLN validation evidence overwrite')
torch.npu.set_device(0);torch.set_num_threads(16)
r={'cases':[],'boundaries':[],'scope':'Actual first-step rank0 indexed FP16 rows; current eager contract, not official Triton BF16/full-video lossless benchmark'}


def metric(a,b):
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
    e=x.float().double()-y.float().double()
    return {'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)),
            'finite':bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
            'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.float().double().square().mean().sqrt().clamp_min(1e-30)),
            'word_mismatches':int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),'elements':x.numel()}


def timed(fn):
    fn();torch.npu.synchronize();times=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(5):out=fn()
        torch.npu.synchronize();times.append((time.perf_counter()-t)*1000/5)
    return out,{'groups_ms':times,'median_ms':statistics.median(times)}


try:
    with torch.inference_mode():
        names={'indexed_scale_shift_':'mod','indexed_gate':'gate','rms_norm_indexed_scale_shift':'rms_mod','indexed_gate_rms_norm_scale_shift':'gate_rms_mod'}
        for name,kind in names.items():
            for call in ((0,) if kind=='mod' else (0,24,49)):
                d=torch.load(ROOT/'capture'/(name+'_%03d.pt'%call),weights_only=True)
                cpu=d['inputs'];v={k:x.to('npu').contiguous() if isinstance(x,torch.Tensor) else x for k,x in cpu.items()}
                indices=v['indices'];x=v.get('x',v.get('residual'))
                if kind=='mod':
                    eager=lambda:reference.indexed_scale_shift_(x.clone(),v['shift'],v['scale'],indices)
                    fused=lambda:launch(kind,x,indices,shift=v['shift'],scale=v['scale'])
                elif kind=='gate':
                    eager=lambda:reference.indexed_gate(x,v['gate'],v['other'],indices)
                    fused=lambda:launch(kind,x,indices,gate=v['gate'],branch=v['other'])
                elif kind=='rms_mod':
                    eager=lambda:reference.rms_norm_indexed_scale_shift(x,v['weight'],v['shift'],v['scale'],indices,v['eps'])
                    fused=lambda:launch(kind,x,indices,weight=v['weight'],shift=v['shift'],scale=v['scale'],eps=v['eps'])
                else:
                    eager=lambda:reference.indexed_gate_rms_norm_scale_shift(x,v['gate'],v['branch'],v['weight'],v['shift'],v['scale'],indices,v['eps'])
                    fused=lambda:launch(kind,x,indices,gate=v['gate'],branch=v['branch'],weight=v['weight'],shift=v['shift'],scale=v['scale'],eps=v['eps'])
                t=time.perf_counter();fused();torch.npu.synchronize();cold=time.perf_counter()-t
                got,ft=timed(fused);base,et=timed(eager)
                gos=list(got) if isinstance(got,tuple) else [got]
                bases=list(base) if isinstance(base,tuple) else [base]
                caps=list(d['output']) if isinstance(d['output'],tuple) else [d['output']]
                row={'name':name,'call':call,'shape':list(x.shape),'table_rows':v.get('gate',v.get('shift')).shape[0],
                     'cold_build_and_call_s':cold,'vs_same_input_npu_eager':[metric(a,b) for a,b in zip(gos,bases)],
                     'vs_actual_capture':[metric(a,b) for a,b in zip(gos,caps)],
                     'eager_vs_actual_capture':[metric(a,b) for a,b in zip(bases,caps)],'fused_wrapper':ft,'eager_wrapper':et}
                r['cases'].append(row);OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(row),flush=True)
        torch.manual_seed(61)
        for rows in (7,8,9,129):
            width=128
            x=torch.randn(rows,width,dtype=torch.float16,device='npu')
            branch=torch.randn_like(x);weight=torch.randn(width,dtype=x.dtype,device=x.device)
            shift=torch.randn(6,width,dtype=x.dtype,device=x.device);scale=torch.randn_like(shift);gate=torch.randn_like(shift)
            indices=(torch.arange(rows,device=x.device)*5%6).long()
            refs={'mod':lambda:reference.indexed_scale_shift_(x.clone(),shift,scale,indices),
                  'gate':lambda:reference.indexed_gate(x,gate,branch,indices),
                  'rms_mod':lambda:reference.rms_norm_indexed_scale_shift(x,weight,shift,scale,indices,1e-5),
                  'gate_rms_mod':lambda:reference.indexed_gate_rms_norm_scale_shift(x,gate,branch,weight,shift,scale,indices,1e-5)}
            for kind in names.values():
                args={}
                if 'gate' in kind:args.update(gate=gate,branch=branch)
                if 'rms' in kind:args.update(weight=weight)
                if kind!='gate':args.update(shift=shift,scale=scale)
                got=launch(kind,x,indices,**args);ref=refs[kind]();torch.npu.synchronize()
                a=list(got) if isinstance(got,tuple) else [got];b=list(ref) if isinstance(ref,tuple) else [ref]
                row={'rows':rows,'kind':kind,'comparisons':[metric(xx,yy) for xx,yy in zip(a,b)]}
                r['boundaries'].append(row)
        r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
