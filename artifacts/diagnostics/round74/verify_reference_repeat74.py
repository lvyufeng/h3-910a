#!/usr/bin/env python3
"""Post-completion verification of original-BF16 HF language repeat.

Read final tensors only AFTER both write-once producer receipts report complete.
This removes publication races and conditional-check ambiguity from acceptance,
without modifying/replaying either executed reference or washing away failures.
Both producer field schemas are mapped explicitly and full PRE-final-norm outputs
plus every layer's raw tensor identity are required to match.
"""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_repeat01');FIRST=Path('/tmp/h3_round74_hf01')
writer=EvidenceWriter(ROOT/'reference_stability01.json');torch.set_num_threads(8)
r={'scope':__doc__,'completed':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def raw(v):return hashlib.sha256(memoryview(v.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()


def full_proof(a,b):
    assert a.shape==b.shape==(1,2043,5120)and a.dtype==b.dtype==torch.bfloat16
    x,y=a.contiguous().reshape(-1),b.contiguous().reshape(-1);err=0.;ref=0.;mx=0.;different=0;finite=True
    for s in range(0,x.numel(),1048576):
        xx,yy=x[s:s+1048576],y[s:s+1048576];different+=int((xx.view(torch.uint8)!=yy.view(torch.uint8)).sum())
        xd,yd=xx.double(),yy.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;err+=float(e.square().sum());ref+=float(yd.square().sum());mx=max(mx,float(e.abs().max()))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'raw_bits_equal':different==0,'different_bytes':different,
            'finite':finite,'max_abs_error':mx if finite else None,'relative_rms':(err/max(ref,1e-30))**.5 if finite else None,
            'first_sha256':raw(a),'repeat_sha256':raw(b)}


try:
    first=json.loads((FIRST/'hf_reference01.json').read_text());repeat=json.loads((ROOT/'language_repeat01.json').read_text())
    assert first.get('completed')and first.get('full_multimodal_reference_completed')and not first.get('failure')
    assert repeat.get('completed')and not repeat.get('failure')
    assert len(first['language_layers'])==len(repeat['layers'])==50
    r['independent_HF_multimodal_input_sha256']=repeat['independent_HF_multimodal_input_sha256']
    r['first_independent_HF_input_sha256']=first['stages']['hf_multimodal_inputs']['file_sha256']
    assert r['independent_HF_multimodal_input_sha256']==r['first_independent_HF_input_sha256']
    r['all50_layer_raw_identities']=[]
    for a,b in zip(first['language_layers'],repeat['layers']):
        row={'layer':a['layer'],'first_sha256':a['output']['sha256'],'repeat_sha256':b['sha256'],
             'finite':a['output']['finite']and b['finite']}
        assert a['layer']==b['layer']and row['first_sha256']==row['repeat_sha256']and row['finite']
        r['all50_layer_raw_identities'].append(row)
    one=torch.load(FIRST/'hf_selected50.pt',map_location='cpu',weights_only=True)
    two=torch.load(ROOT/'hf_repeat_selected50.pt',map_location='cpu',weights_only=True)
    r['producer_field_schemas']={'first':['pre_final_norm','hf_post_final_norm'],'repeat':['pre_final_norm','post_final_norm']}
    r['pre_final_norm_repeat']=full_proof(one['pre_final_norm'],two['pre_final_norm'])
    r['post_final_norm_repeat']=full_proof(one['hf_post_final_norm'],two['post_final_norm'])
    for p in(r['pre_final_norm_repeat'],r['post_final_norm_repeat']):assert p['raw_bits_equal']and p['finite']and p['max_abs_error']==0
    r['pre_final_norm_matches_layer49_identity']=r['pre_final_norm_repeat']['first_sha256']==first['language_layers'][-1]['output']['sha256']
    assert r['pre_final_norm_matches_layer49_identity']
    r['original_conditional_check_was_present']='reference_repeat_stability'in repeat
    r['posthoc_full_reference_repeat_accepted']=True;r['completed']=True
except BaseException as e:r['failure']={'type':type(e).__name__,'message':str(e)};raise
finally:writer.write(r)
