#!/usr/bin/env python3
"""Actual same-input full-matrix FP32 attention geometry probes, not automatic integration.

Zeros added to head width are mathematically neutral but may alter NPU GEMM
accumulation/dispatch. Therefore QK, scaled score, softmax, PV and BF16 outputs
must independently match full raw bits. Padding allocation/copies stay timed.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io73 import EvidenceWriter

ROOT=Path('/tmp/h3_round73_geometry01')
CAP=Path('/tmp/h3_round73_attention01')
writer=EvidenceWriter(ROOT/'geometry01.json')
torch.npu.set_device(0);torch.set_num_threads(8)
r={'scope':__doc__,'calls':[], 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'runtime_math_flags_before':{'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32}}
MODES={'native':(72,72),'qk80':(80,72),'qk128':(128,72),'pv80':(72,80),'pv128':(72,128),'both128':(128,128),
       'chunk1008':(72,72),'chunk2016':(72,72),'no_contiguous':(72,72)}


def raw(value):
    a=value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(a.view(torch.uint8).numpy())).hexdigest()


def metric(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
    xf,yf=x.reshape(-1),y.reshape(-1)
    bits=True;finite=True;different=0;maximum=0.;error2=0.;ref2=0.
    word=torch.uint16 if x.element_size()==2 else torch.uint32
    for start in range(0,x.numel(),1048576):
        aa,bb=xf[start:start+1048576],yf[start:start+1048576]
        mismatch=aa.view(word).to(torch.int64)!=bb.view(word).to(torch.int64)
        n=int(mismatch.sum());different+=n;bits=bits and n==0
        af,bf=aa.double(),bb.double()
        valid=bool(torch.isfinite(af).all() and torch.isfinite(bf).all());finite=finite and valid
        if valid:
            error=af-bf;maximum=max(maximum,float(error.abs().max()));error2+=float(error.square().sum());ref2+=float(bf.square().sum())
    return {'shape':list(x.shape),'dtype':str(x.dtype),'elements':x.numel(),'raw_bits_equal':bits,'different_words':different,'finite':finite,
            'max_abs_error':maximum if finite else None,'relative_rms':(error2/max(ref2,1e-30))**.5 if finite else None}


def layout(value):
    return {'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype),
            'physical_format':torch_npu.get_npu_format(value),'storage_offset':value.storage_offset()}


def stages(q,k,v,mode):
    if mode.startswith('chunk'):
        length=int(mode[5:]);parts=[]
        for start in range(0,q.shape[-2],length):
            parts.append(stages(q[...,start:start+length,:],k,v,'native'))
        return {name:torch.cat([part[name] for part in parts],dim=-2) for name in parts[0]}
    qf,kf,vf=(x.float() if mode=='no_contiguous' else x.float().contiguous() for x in (q,k,v))
    qwidth,vwidth=MODES[mode]
    if qwidth!=72:qf,kf=F.pad(qf,(0,qwidth-72)),F.pad(kf,(0,qwidth-72))
    unscaled=torch.matmul(qf,kf.transpose(-2,-1))
    scores=unscaled*(72**-0.5)
    probability=torch.softmax(scores,dim=-1)
    if vwidth!=72:vf=F.pad(vf,(0,vwidth-72))
    partial=torch.matmul(probability,vf)
    if vwidth!=72:partial=partial[...,:72].contiguous()
    output=partial.to(q.dtype)
    return {'unscaled_qk':unscaled,'scores':scores,'probabilities':probability,'partial_pv':partial,'output':output}


def forward(q,k,v,mode):
    if mode.startswith('chunk'):
        length=int(mode[5:])
        return torch.cat([forward(q[...,start:start+length,:],k,v,'native') for start in range(0,q.shape[-2],length)],dim=-2)
    qf,kf,vf=(x.float() if mode=='no_contiguous' else x.float().contiguous() for x in (q,k,v))
    qwidth,vwidth=MODES[mode]
    if qwidth!=72:qf,kf=F.pad(qf,(0,qwidth-72)),F.pad(kf,(0,qwidth-72))
    scores=torch.matmul(qf,kf.transpose(-2,-1))*(72**-0.5)
    p=torch.softmax(scores,dim=-1)
    if vwidth!=72:vf=F.pad(vf,(0,vwidth-72))
    out=torch.matmul(p,vf)
    if vwidth!=72:out=out[...,:72].contiguous()
    return out.to(q.dtype)


def timed(q,k,v,mode):
    torch.npu.synchronize();start=time.perf_counter()
    output=forward(q,k,v,mode)
    torch.npu.synchronize();seconds=time.perf_counter()-start
    return output,seconds


try:
    with torch.inference_mode():
        for index in (0,17,33,53):
            path=CAP/f'attention_call{index:02d}.pt';meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['same_input_original_bits_equal'] and hashlib.sha256(path.read_bytes()).hexdigest()==meta['file_sha256']
            d=torch.load(path,map_location='cpu',weights_only=True)
            inputs=[];reconstructed=[]
            for name in ('query','key','value'):
                info=meta['input_layouts'][name]
                source=torch.full((info['storage_bytes']//2,),3.,dtype=torch.bfloat16,device='npu:0')
                view=source.as_strided(tuple(info['shape']),tuple(info['stride']),info['storage_offset'])
                view.copy_(d[name]);assert raw(view)==raw(d[name])
                assert list(view.stride())==info['stride'] and view.storage_offset()==info['storage_offset']
                inputs.append(view);reconstructed.append(layout(view))
            q,k,v=inputs
            before=[raw(value) for value in inputs]
            reference=stages(q,k,v,'native')
            capture_proof=metric(reference['output'],d['output']);assert capture_proof['raw_bits_equal'] and capture_proof['finite']
            stable=stages(q,k,v,'native')
            native_repeat={name:metric(stable[name],reference[name]) for name in reference}
            assert all(p['raw_bits_equal'] and p['finite'] for p in native_repeat.values())
            del stable
            row={'call':index,'input_layouts_reconstructed':reconstructed,'native_output_vs_capture':capture_proof,
                 'native_repeat_full_stages':native_repeat,'cases':{}}
            r['calls'].append(row)
            for mode in MODES:
                try:
                    candidate=stages(q,k,v,mode)
                    proofs={name:metric(candidate[name],reference[name]) for name in reference}
                    format_equal=layout(candidate['output'])==layout(reference['output'])
                    del candidate
                    first,first_s=timed(q,k,v,mode)
                    case={'full_stage_proofs':proofs,'output_native_layout_equal':format_equal,
                          'all_stages_exact':all(p['raw_bits_equal'] and p['finite'] for p in proofs.values()),
                          'first_s_excluded':first_s,'warm':[]}
                    for _ in range(3):
                        output,seconds=timed(q,k,v,mode)
                        case['warm'].append({'seconds':seconds,'output_vs_native':metric(output,reference['output'])})
                    case['median_s']=statistics.median(item['seconds'] for item in case['warm'])
                    row['cases'][mode]=case
                    writer.write(r)
                    print(json.dumps({'call':index,'mode':mode,'median_s':case['median_s'],'all_stage_bits':case['all_stages_exact'],
                                      'qk_bits':proofs['unscaled_qk']['raw_bits_equal'],'pv_bits':proofs['partial_pv']['raw_bits_equal'],
                                      'output_bits':proofs['output']['raw_bits_equal']}),flush=True)
                except Exception as exc:
                    row['cases'][mode]={'failure':{'type':type(exc).__name__,'message':str(exc)}}
                    writer.write(r)
                    print(json.dumps({'call':index,'mode':mode,'failure':str(exc)}),flush=True)
            row['input_immutable']=before==[raw(value) for value in inputs];assert row['input_immutable']
            writer.write(r)
            del reference,inputs,q,k,v
    r['runtime_math_flags_after']={'matmul_allow_hf32':torch.npu.matmul.allow_hf32,'conv_allow_hf32':torch.npu.conv.allow_hf32}
    assert r['runtime_math_flags_before']==r['runtime_math_flags_after']
    r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)}
    raise
finally:
    writer.write(r)
