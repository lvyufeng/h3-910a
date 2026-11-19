#!/usr/bin/env python3
"""Compare the 910A FP16 encoder against 4090 FP16 and original-BF16 references.

GPU FP16 vs GPU BF16 isolates the inherent loss of FP16 conversion on the *same*
undistilled official weights; 910A FP16 vs GPU FP16 isolates the 910A stack itself.
No reference is cast to FP16 to flatter the comparison, and no tolerance is declared.
"""
from pathlib import Path
import hashlib,json


def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def metric(candidate,reference,name):
    import torch
    a=candidate.detach().float().double();b=reference.detach().float().double()
    assert a.shape==b.shape==(1,2043,5120),(a.shape,b.shape)
    both=torch.isfinite(a)&torch.isfinite(b)
    d=(a-b)[both]
    return {'name':name,'candidate_dtype':str(candidate.dtype),'reference_dtype':str(reference.dtype),
            'nonfinite_candidate':int((~torch.isfinite(a)).sum()),'nonfinite_reference':int((~torch.isfinite(b)).sum()),
            'compared_elements':int(both.sum()),
            'relative_rms':float(d.square().mean().sqrt()/b[both].square().mean().sqrt()),
            'max_abs_error':float(d.abs().max()),
            'relative_max_abs_error':float(d.abs().max()/b[both].abs().max()),
            'numerical_tolerance':None,'declared_pass':False}


def subset_metric(candidate,reference,mask,name):
    import torch
    flat_a=candidate.reshape(-1,5120).float().double();flat_b=reference.reshape(-1,5120).float().double()
    sel=mask.reshape(-1).bool();a=flat_a[sel];b=flat_b[sel];d=a-b
    return {'name':name,'rows':int(sel.sum()),
            'relative_rms':float(d.square().mean().sqrt()/b.square().mean().sqrt()),
            'max_abs_error':float(d.abs().max())}


def main():
    import torch
    root=Path('/home/mseco/h3_gpu_reference114')
    receipt=json.loads((root/'GPU_encoder_reference114.json').read_text())
    assert receipt['completed'],(receipt.get('failure'),list(receipt.get('cases',{})))
    gpu={}
    for label in('BF16','FP16'):
        meta=receipt['cases'][label]['saved_output']
        path=Path(meta['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['file_SHA256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        gpu[label]=payload['pre_final_norm']
        assert sha(gpu[label])==receipt['cases'][label]['pre_final_norm']['sha256']
    npu_cases={}
    for name in('corrected_VisionTP8_native_text','native'):
        path=root/('NPU_selected50_%s.pt'%name)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        mask=payload['visual_pos_masks']
        npu_cases[name]={'tensor':payload['selected50'],'visual_mask':mask}
    reference_mask=npu_cases['corrected_VisionTP8_native_text']['visual_mask']
    out={'scope':__doc__,'completed':False,'whole_goal_complete':False,
         'full_encoder_precision_accepted':False,'numerical_tolerance':None,
         'gpu_reference':{'BF16':receipt['cases']['BF16']['pre_final_norm'],
                          'FP16':receipt['cases']['FP16']['pre_final_norm'] if 'FP16'in receipt['cases']else None},
         'comparisons':{}}
    if 'FP16' in receipt['cases']:
        out['comparisons']['GPU_FP16_vs_GPU_BF16']=metric(gpu['FP16'],gpu['BF16'],'inherent_FP16_loss')
        out['comparisons']['GPU_FP16_vs_GPU_BF16_visual']=subset_metric(gpu['FP16'],gpu['BF16'],reference_mask,'inherent_FP16_loss_visual_rows')
        out['comparisons']['GPU_FP16_vs_GPU_BF16_text']=subset_metric(gpu['FP16'],gpu['BF16'],~reference_mask.bool(),'inherent_FP16_loss_text_rows')
    for name,payload in npu_cases.items():
        key='NPU_%s'%name
        if 'FP16' in receipt['cases']:
            out['comparisons'][key+'_vs_GPU_FP16']=metric(payload['tensor'],gpu['FP16'],'910A_vs_same_precision_reference')
            out['comparisons'][key+'_vs_GPU_FP16_visual']=subset_metric(payload['tensor'],gpu['FP16'],reference_mask,'910A_vs_GPU_FP16_visual')
            out['comparisons'][key+'_vs_GPU_FP16_text']=subset_metric(payload['tensor'],gpu['FP16'],~reference_mask.bool(),'910A_vs_GPU_FP16_text')
        out['comparisons'][key+'_vs_GPU_BF16']=metric(payload['tensor'],gpu['BF16'],'910A_FP16_vs_original_BF16')
    out['completed']='FP16' in receipt['cases']
    dest=root/'GPU_vs_NPU_encoder114.json'
    tmp=dest.with_name(dest.name+'.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(dest)
    print(json.dumps({k:{'relative_rms':v.get('relative_rms'),'max_abs_error':v.get('max_abs_error'),
                         'candidate_dtype':v.get('candidate_dtype'),'reference_dtype':v.get('reference_dtype')}
                      for k,v in out['comparisons'].items()},indent=2))


if __name__=='__main__':main()