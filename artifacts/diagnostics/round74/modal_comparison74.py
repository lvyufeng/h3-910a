#!/usr/bin/env python3
"""Independent-stage HF/experimental comparison, modal split and outlier diagnostics.

This consumes newly generated independent HF outputs only after their write-once
manifest appears. It never uses experimental features as HF inputs,never changes
weights/math and does not label measured numerical differences a quality pass.
"""
import hashlib
import json
from pathlib import Path
import torch
from evidence_io74 import EvidenceWriter

ROOT=Path('/tmp/h3_round74_hf01');CAP=Path('/tmp/h3_round74_capture01')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
EXP=Path('/tmp/h3_round70_components01/fresh_encoder_reference.pt')
writer=EvidenceWriter(ROOT/'modal_comparison01.json')
torch.set_num_threads(8)
r={'scope':__doc__,'comparisons':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'independent_hf_reference_used':True,'reference_was_fp32_linear_adapter':False,
   'full_multimodal_hf_identity_accepted':False,'official_goal_complete':False,'production_modified':False}


def metrics(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    x,y=a.contiguous().reshape(-1),b.contiguous().reshape(-1)
    err=0.;ref=0.;maxabs=0.;different=0;finite=True
    for start in range(0,x.numel(),1048576):
        xx,yy=x[start:start+1048576],y[start:start+1048576]
        different+=int((xx.view(torch.uint8)!=yy.view(torch.uint8)).sum())
        xd,yd=xx.double(),yy.double();valid=bool(torch.isfinite(xd).all()and torch.isfinite(yd).all());finite=finite and valid
        if valid:
            e=xd-yd;err+=float(e.square().sum());ref+=float(yd.square().sum());maxabs=max(maxabs,float(e.abs().max()))
    return {'shape':list(a.shape),'dtype':str(a.dtype),'elements':a.numel(),'raw_bits_equal':different==0,'different_bytes':different,
            'finite':finite,'max_abs_error':maxabs if finite else None,'relative_rms':(err/max(ref,1e-30))**.5 if finite else None}


def modal(a,b,visual):
    assert a.shape==b.shape==(1,2043,5120)
    return {'full':metrics(a,b),'text_positions':metrics(a[~visual],b[~visual]),'visual_positions':metrics(a[visual],b[visual])}


def outliers(a,b,visual):
    if not bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all()):
        return {'finite':False,'outlier_statistics':None,'nonfinite_tensor_acceptance_rejected':True}
    e=(a.double()-b.double()).square()[0];baseline=b.double()[0]
    channels=e.sum(0);tokens=e.sum(1)
    topc=torch.topk(channels,16);topt=torch.topk(tokens,16)
    err=float(e.sum())
    return {'finite':True,'total_squared_error':err,'top16_channels':topc.indices.tolist(),'top16_channel_error_fraction':float(topc.values.sum())/max(err,1e-30),
            'top16_tokens':topt.indices.tolist(),'top16_token_error_fraction':float(topt.values.sum())/max(err,1e-30),
            'top16_token_is_visual':[bool(visual[0,i])for i in topt.indices],
            'reference_max_abs':float(baseline.abs().max()),'experimental_max_abs':float(a.double().abs().max()),
            'visual_error_fraction':float(e[visual[0]].sum())/max(err,1e-30)}


try:
    receipt=json.loads((ROOT/'hf_reference01.json').read_text())
    assert receipt.get('full_multimodal_reference_completed')and receipt.get('completed')and not receipt.get('failure')
    assert receipt['reference_is_original_bf16_HF']and not receipt['shared_native_visual_features_used']and not receipt['fp32_linear_adapter_used']
    r['independent_hf_reference_used']=bool(receipt['reference_is_original_bf16_HF']and not receipt['shared_native_visual_features_used'])
    r['reference_was_fp32_linear_adapter']=bool(receipt['fp32_linear_adapter_used'])
    exp=torch.load(EXP,map_location='cpu',weights_only=True)
    vision=torch.load(ROOT/'hf_vision.pt',map_location='cpu',weights_only=True)
    textinput=torch.load(ROOT/'hf_multimodal_inputs.pt',map_location='cpu',weights_only=True)
    final=torch.load(ROOT/'hf_selected50.pt',map_location='cpu',weights_only=True)
    assert len(exp['vision_output'][1])==len(vision['deepstack'])==len(textinput['deepstack_visual_embeds'])==3
    visual=textinput['visual_pos_masks']
    r['token_counts']={'total':visual.numel(),'visual':int(visual.sum()),'text':int((~visual).sum())}
    assert r['token_counts']=={'total':2043,'visual':2016,'text':27}
    r['comparisons']['vision_merged']=metrics(exp['vision_output'][0],vision['merged'])
    r['comparisons']['vision_deepstack']=[metrics(a,b)for a,b in zip(exp['vision_output'][1],vision['deepstack'])]
    r['comparisons']['text_positions']=metrics(exp['text_inputs']['positions'],textinput['position_ids'])
    r['comparisons']['visual_mask']=metrics(exp['text_inputs']['visual_pos_masks'],visual)
    r['comparisons']['text_input']=modal(exp['text_inputs']['inputs_embeds'],textinput['inputs_embeds'],visual)
    r['comparisons']['selected50']=modal(exp['complete_output'][None],final['pre_final_norm'],visual)
    r['selected50_outliers']=outliers(exp['complete_output'][None],final['pre_final_norm'],visual)
    r['hf_final_norm_boundary']=metrics(final['pre_final_norm'],final['hf_post_final_norm'])
    r['final_norm_difference_is_excluded_from_h3_comparison']=True
    r['full_multimodal_hf_identity_accepted']=r['comparisons']['selected50']['full']['raw_bits_equal']and r['comparisons']['selected50']['full']['finite']
    r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:writer.write(r)
