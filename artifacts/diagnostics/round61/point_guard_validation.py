#!/usr/bin/env python3
"""Rejection guards, in-place adapter and finite FP16 point arithmetic sweep."""
import json
import os
from pathlib import Path
import torch
import torch_npu
from vllm_omni.diffusion.layers import indexed_modulation as ref
from adaln_vector61 import launch, PreparedIndices

ROOT=Path('/tmp/h3_round61_adaln01');OUT=ROOT/'point_guard_validation04.json'
if OUT.exists():raise RuntimeError('Refuse finite point contract proof overwrite')
torch.npu.set_device(0);torch.set_num_threads(8)
r={'guards':{},'finite_sweep':[],'scope':'Finite FP16 point inputs across signed-zero/extreme factors; invalid call guards; not nonfinite payload/Norm/general concurrent service certification'}


def metric(a,b):
    x,y=a.cpu().contiguous(),b.cpu().contiguous()
    different=x.view(torch.int16).ne(y.view(torch.int16))
    return {'raw_bits_equal':not bool(different.any()),'word_mismatches':int(different.sum()),'elements':x.numel(),
            'output_nan_count':int(torch.isnan(x.float()).sum()),'output_inf_count':int(torch.isinf(x.float()).sum()),
            'examples':list(zip(x.view(torch.int16)[different][:6].tolist(),y.view(torch.int16)[different][:6].tolist()))}


def reject(name,fn):
    try:fn()
    except (AssertionError,ValueError):r['guards'][name]=True
    else:raise RuntimeError(name+' was not rejected')


try:
    with torch.inference_mode():
        idx=torch.arange(8,device='npu').long()%6;prepared=PreparedIndices(idx,6)
        x=torch.ones(8,128,dtype=torch.float16,device='npu');gate=torch.ones(6,128,dtype=x.dtype,device=x.device)
        offset=torch.ones(6,129,dtype=x.dtype,device=x.device)[:,1:]
        reject('offset_pointer_alignment',lambda:launch('gate',x,prepared,gate=offset,branch=x))
        stride=torch.ones(6,129,dtype=x.dtype,device=x.device)[:,:128]
        reject('row_stride_alignment',lambda:launch('gate',x,prepared,gate=stride,branch=x))
        torch.npu.set_device(1)
        reject('different_current_device',lambda:launch('gate',x,prepared,gate=gate,branch=x))
        torch.npu.set_device(0)
        snap=PreparedIndices(idx,6)
        snap.tensor.add_(1);reject('prepared_mutation',lambda:snap.checked(8,6,x.device))
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):reject('cross_stream_snapshot',lambda:prepared.checked(8,6,x.device))
        reject('negative_index',lambda:PreparedIndices(torch.tensor([-1],device='npu'),6))
        reject('index_beyond_table',lambda:PreparedIndices(torch.tensor([6],device='npu'),6))
        # In-place adapter must return the original tensor and update its alias.
        shift=torch.zeros_like(gate);scale=torch.ones_like(gate)
        expected=ref.indexed_scale_shift_(x.clone(),shift,scale,idx)
        pointer=x.data_ptr();alias=x.view_as(x)
        updated=x.copy_(launch('mod',x,prepared,shift=shift,scale=scale))
        torch.npu.synchronize()
        r['inplace']={'returned_same_tensor':updated is x,'same_pointer':x.data_ptr()==pointer,'alias_bits_equal':metric(alias,expected)['raw_bits_equal']}
        assert all(r['inplace'].values())
        # Every finite half word; number of elements is divisible by128.
        words=torch.arange(65536,dtype=torch.int32)
        finite=words[(words&0x7c00)!=0x7c00].to(torch.int16).view(torch.float16).contiguous()
        assert finite.numel()%128==0
        rows=finite.numel()//128
        inputs=finite.reshape(rows,128).to('npu')
        indices=torch.arange(rows,device='npu').long()%6;metadata=PreparedIndices(indices,6)
        for factor in (-65504.,-2.,-1.,-0.,0.,0.5,1.,2.,65504.):
            gate=torch.full((6,128),factor,dtype=torch.float16,device='npu')
            base=ref.indexed_gate(inputs,gate,inputs,indices)
            got=launch('gate',inputs,metadata,gate=gate,branch=inputs)
            torch.npu.synchronize()
            row={'kind':'gate','factor':factor,**metric(got,base)};r['finite_sweep'].append(row)
        for scale_value in (-65504.,-2.,-1.,-0.,0.,0.5,1.,65504.):
            scale=torch.full((6,128),scale_value,dtype=torch.float16,device='npu')
            for shift_value in (-0.,0.,1.):
                shift=torch.full_like(scale,shift_value)
                base=ref.indexed_scale_shift_(inputs.clone(),shift,scale,indices)
                got=launch('mod',inputs,metadata,shift=shift,scale=scale)
                torch.npu.synchronize()
                row={'kind':'mod','scale':scale_value,'shift':shift_value,**metric(got,base)};r['finite_sweep'].append(row)
        r['all_finite_sweep_bits_equal']=all(z['raw_bits_equal'] for z in r['finite_sweep'])
    # Create ordinary differentiable tensors outside inference mode. Guard all
    # operand gradients, not merely x.requires_grad.
    with torch.inference_mode(False),torch.enable_grad():
        x=torch.ones(8,128,dtype=torch.float16,device='npu');branch=torch.ones_like(x)
        gate=torch.ones(6,128,dtype=x.dtype,device=x.device,requires_grad=True)
        idx=torch.arange(8,device='npu').long()%6;metadata=PreparedIndices(idx,6)
        reject('gate_gradient',lambda:launch('gate',x,metadata,gate=gate,branch=branch))
        gate=gate.detach();branch.requires_grad_()
        reject('branch_gradient',lambda:launch('gate',x,metadata,gate=gate,branch=branch))
        branch=branch.detach();weight=torch.ones(128,dtype=x.dtype,device=x.device,requires_grad=True)
        shift=torch.zeros_like(gate);scale=torch.ones_like(gate)
        reject('norm_weight_gradient',lambda:launch('rms_mod',x,metadata,weight=weight,shift=shift,scale=scale))
    r['completed']=True
except BaseException as exc:r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
