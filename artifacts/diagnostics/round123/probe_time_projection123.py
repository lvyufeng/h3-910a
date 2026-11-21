#!/usr/bin/env python3
"""Actual first-step time-embedding FP32 projection probe with exact TP2 math.

Runs only on GPU/NPU; no CPU model reference. Preserves original FP32 time weights,
sin/cos FP32 input, original SiLU, and TP2 split/gather/sum order. The NPU candidate
uses the same four Half-operand FP32-output residual guard tested on patch inputs.
This is a two-timestep leaf diagnostic, not an end-to-end speed/quality acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['cuda','npu'],required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reference')
    args=parser.parse_args()
    import torch
    if args.backend=='npu':
        import torch_npu
        torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False
        from fp32_patch_residual_guard123 import ResidualFP32Projection123
    else:
        torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    from safetensors import safe_open
    torch.set_num_threads(4)
    root=Path(args.root);checkpoint=Path(args.checkpoint)
    output=root/f'{args.backend}_time_projection_probe123.json'
    assert not output.exists()
    device=torch.device(args.backend+':0')
    index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    params={};identities={}
    for stem in ['proj_in','proj_out']:
        for suffix in ['weight','bias']:
            name=f'time_embedder.{stem}.{suffix}'
            with safe_open(str(checkpoint/index[name]),framework='pt',device='cpu') as file:
                value=file.get_tensor(name).float().contiguous()
            identities[name]={'shape':list(value.shape),'original_value_FP32_sha256':hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}
            params[(stem,suffix)]=value.to(device)
    w0,b0=params[('proj_in','weight')],params[('proj_in','bias')]
    w1,b1=params[('proj_out','weight')],params[('proj_out','bias')]
    width=w0.shape[0]//2
    layers0=[SimpleNamespace(weight=w0[i*width:(i+1)*width].contiguous()) for i in range(2)]
    layers1=[SimpleNamespace(weight=w1[:,i*width:(i+1)*width].contiguous()) for i in range(2)]
    b0s=[b0[i*width:(i+1)*width].contiguous() for i in range(2)]
    t=torch.tensor([0.,0.999],dtype=torch.float32,device=device)
    freq=torch.exp(-math.log(10000.)*torch.arange(128,dtype=torch.float32,device=device)/128)
    angles=t[:,None]*freq[None]
    t_freq=torch.cat([torch.cos(angles),torch.sin(angles)],dim=-1)
    with torch.inference_mode():
        projected=torch.cat([torch.nn.functional.linear(t_freq,layer.weight,b0s[i]) for i,layer in enumerate(layers0)],dim=-1)
        hidden=torch.nn.functional.silu(projected)
        partials=[torch.nn.functional.linear(hidden[:,i*width:(i+1)*width].contiguous(),layer.weight) for i,layer in enumerate(layers1)]
        output_native=partials[0]+partials[1]+b1
    tensors={'t_freq':t_freq.cpu().contiguous(),'projection_in_output':projected.cpu().contiguous(),
             'projection_out_input':hidden.cpu().contiguous(),'time_embedding':output_native.cpu().contiguous()}
    result={'scope':__doc__,'completed':False,'backend':args.backend,'original_FP32_parameter_identities':identities,
            'TP2_local_projection_width':width,'numeric_tolerance':None,'full_model_or_speed_accepted':False,
            'CPU_model_inference_executed':False,'native_output_finite':bool(torch.isfinite(output_native).all())}
    if args.backend=='cuda':
        path=root/'GPU_time_projection_reference123.pt'
        with path.open('xb') as file:torch.save(tensors,file)
        result['reference_path']=str(path);result['reference_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        reference=torch.load(args.reference,map_location='cpu',weights_only=True)
        def metric(a,b):
            a=a.double();b=b.double();d=a-b
            return {'relative_RMS':float(d.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-30)),
                    'max_abs_error':float(d.abs().max()),'finite':bool(torch.isfinite(a).all()),'shape':list(a.shape)}
        result['native_stage_errors_vs_GPU']={name:metric(value,reference[name]) for name,value in tensors.items()}
        m0s=[ResidualFP32Projection123() for _ in range(2)]
        m1s=[ResidualFP32Projection123() for _ in range(2)]
        with torch.inference_mode():
            projected_fixed=torch.cat([m0s[i].apply(layer,t_freq,b0s[i]) for i,layer in enumerate(layers0)],dim=-1)
            hidden_fixed=torch.nn.functional.silu(projected_fixed)
            parts_fixed=[m1s[i].apply(layer,hidden_fixed[:,i*width:(i+1)*width].contiguous()) for i,layer in enumerate(layers1)]
            out_fixed=parts_fixed[0]+parts_fixed[1]+b1
            teacher_hidden=reference['projection_out_input'].to(device)
            teacher_native_parts=[torch.nn.functional.linear(teacher_hidden[:,i*width:(i+1)*width].contiguous(),layer.weight) for i,layer in enumerate(layers1)]
            teacher_fixed_parts=[m1s[i].apply(layer,teacher_hidden[:,i*width:(i+1)*width].contiguous()) for i,layer in enumerate(layers1)]
        result['residual_GEMM_stage_errors_vs_GPU']={
            'projection_in_output':metric(projected_fixed.cpu(),reference['projection_in_output']),
            'projection_out_input':metric(hidden_fixed.cpu(),reference['projection_out_input']),
            'time_embedding':metric(out_fixed.cpu(),reference['time_embedding'])}
        result['GPU_hidden_teacher_forced_row_projection']={
            'native':metric((teacher_native_parts[0]+teacher_native_parts[1]+b1).cpu(),reference['time_embedding']),
            'residual_GEMM':metric((teacher_fixed_parts[0]+teacher_fixed_parts[1]+b1).cpu(),reference['time_embedding'])}
    result['completed']=True
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
