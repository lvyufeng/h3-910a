#!/usr/bin/env python3
"""Cross-check fresh HF vision reference against retained independent Round54 HF vision.

This is a comparison of two independently executed full HF BF16 vision towers
using the same original pixel/grid inputs,not a re-execution in this script.
The retained baseline's exact file/source identities are reported rather than
calling an unproven cache a new compute run. No weights or outputs are modified.
"""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_hf01');OLD=Path('/tmp/h3_round54_capture01')
writer=EvidenceWriter(ROOT/'vision_repeat01.json');torch.set_num_threads(8)
r={'scope':__doc__,'fresh_reference':{},'retained_reference':{},'comparisons':{},'completed':False}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def proof(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.contiguous(),b.contiguous()
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),
            'raw_bits_equal':bool(torch.equal(x.view(torch.uint8),y.view(torch.uint8))),
            'finite':bool(torch.isfinite(x.float()).all()and torch.isfinite(y.float()).all()),
            'max_abs_error':float((x.double()-y.double()).abs().max()),
            'fresh_tensor_sha256':hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest(),
            'retained_tensor_sha256':hashlib.sha256(memoryview(y.view(torch.uint8).numpy())).hexdigest()}


try:
    baseline=json.loads((OLD/'hf_vision_reference.json').read_text())
    assert baseline.get('completed')and not baseline.get('failure')and baseline['parameter_tensors']==351
    newer=torch.load(ROOT/'hf_vision.pt',map_location='cpu',weights_only=True)
    older=torch.load(OLD/'hf_vision_outputs.pt',map_location='cpu',weights_only=True)
    for name,path in(('fresh_reference',ROOT/'hf_vision.pt'),('retained_reference',OLD/'hf_vision_outputs.pt')):
        r[name]={'path':str(path),'file_sha256':sha(path),'file_bytes':path.stat().st_size}
    r['retained_reference_receipt']={'path':str(OLD/'hf_vision_reference.json'),'sha256':sha(OLD/'hf_vision_reference.json')}
    source=OLD/'reference_hf_vision.py'
    if source.is_file():r['retained_executed_source']={'path':str(source),'sha256':sha(source)}
    r['comparisons']['merged']=proof(newer['merged'],older['merged'])
    assert len(newer['deepstack'])==len(older['deepstack'])==3
    r['comparisons']['deepstack']=[proof(a,b)for a,b in zip(newer['deepstack'],older['deepstack'])]
    positions=torch.load(ROOT/'hf_multimodal_inputs.pt',map_location='cpu',weights_only=True)['position_ids']
    r['comparisons']['positions']=proof(positions,older['positions'])
    r['all_full_vision_outputs_bits_equal']=all(p['raw_bits_equal']and p['finite']and p['max_abs_error']==0 for p in [r['comparisons']['merged'],*r['comparisons']['deepstack'],r['comparisons']['positions']])
    assert r['all_full_vision_outputs_bits_equal']
    r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:writer.write(r)
