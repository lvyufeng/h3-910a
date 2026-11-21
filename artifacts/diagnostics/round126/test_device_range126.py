#!/usr/bin/env python3
"""Actual 910A unit proof for exact device range selection; no model weights."""
import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path
from types import SimpleNamespace


def from_bits(word):
    return struct.unpack('<f',struct.pack('<I',word))[0]


def as_bits(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--label',required=True)
    args=p.parse_args()
    root=Path(args.root).resolve();assert root==Path('/tmp/h3_perf12601')
    assert '/' not in args.label
    destination=root/(args.label+'.json');assert not destination.exists()
    import torch
    import torch_npu
    from device_range126 import DeviceRangeState126, ExactDeviceRangeLinear126
    from h3_replay123.transformer import _MiniMaxH3HalfInputFP32OutputLinearMethod
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    state=DeviceRangeState126();candidate=ExactDeviceRangeLinear126(state)
    rng=random.Random(126)
    words={0,1,0x007fffff,0x00800000,0x7f7fffff}
    for i in range(114):
        b=as_bits(32752.0*(2.0**i));words.update((b-1,b,b+1))
    words.update(rng.randrange(0x7f800000) for _ in range(1024))
    words=sorted(words);values=[from_bits(b) for b in words]
    peaks=torch.tensor(values,dtype=torch.float32,device='npu:0')
    with torch.inference_mode():
        selected=torch.stack([candidate.select(peaks[i]) for i in range(len(words))]).cpu().tolist()
    failures=[]
    for word,peak,scale in zip(words,values,selected):
        e=max(0,math.ceil(math.log2(peak/32752.0))) if peak>32752.0 else 0
        if scale!=2.0**e:failures.append({'word':word,'peak':peak,'reference':2.0**e,'candidate':scale})
    original=_MiniMaxH3HalfInputFP32OutputLinearMethod()
    generator=torch.Generator().manual_seed(126)
    leaves=[]
    with torch.inference_mode():
        for label,peak in [('small',1.0),('threshold_below',from_bits(as_bits(32752.0)-1)),
                           ('threshold_equal',32752.0),('threshold_above',from_bits(as_bits(32752.0)+1)),
                           ('high',2.0**25),('higher',2.0**50)]:
            x=torch.rand((17,64),generator=generator,dtype=torch.float32)*peak
            x[0,0]=peak;x=x.to('npu:0')
            weight=(torch.rand((33,64),generator=generator,dtype=torch.float32)*0.2-0.1).half().to('npu:0')
            bias=(torch.rand((33,),generator=generator,dtype=torch.float32)*0.2-0.1).to('npu:0')
            layer=SimpleNamespace(weight=weight)
            state.begin();a=original.apply(layer,x,bias);b=candidate.apply(layer,x,bias);state.end()
            ac=a.cpu().contiguous();bc=b.cpu().contiguous()
            words_equal=torch.equal(ac.view(torch.int32),bc.view(torch.int32))
            leaves.append({'label':label,'shape':list(ac.shape),'elements':ac.numel(),
                           'bit_identical':words_equal,'reference_finite':bool(torch.isfinite(ac).all()),
                           'candidate_finite':bool(torch.isfinite(bc).all())})
    result={'scope':__doc__,'completed':True,'CPU_model_inference':False,'model_weights_loaded':False,
            'positive_finite_peak_words_tested':len(words),'selector_failures':failures,
            'synthetic_gemm_leaves':leaves,
            'verification_passed':not failures and all(r['bit_identical'] for r in leaves),
            'real_H3_leaf_proof_still_required':True}
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    raise SystemExit(0 if result['verification_passed'] else 1)

if __name__=='__main__':main()
