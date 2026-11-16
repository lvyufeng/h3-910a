#!/usr/bin/env python3
"""Actual AdaLN chunk-view strides and prepared-metadata guard verification."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from vllm_omni.diffusion.layers import indexed_modulation as ref
from adaln_vector61 import launch, PreparedIndices

ROOT=Path('/tmp/h3_round61_adaln01')
OUT=ROOT/'point_stride_validation03.json'
if OUT.exists():raise RuntimeError('Refuse strided point proof overwrite')
torch.npu.set_device(0)
r={'cases':[], 'guards':{}, 'scope':'True AdaLN chunk row strides and table offsets, complete wrappers on actual first-step rows; not whole generation or Norm-lossless evidence'}


def bits(a,b):
    x,y=a.detach().cpu().contiguous(),b.detach().cpu().contiguous()
    e=x.float().double()-y.float().double()
    return {'raw_bits_equal':torch.equal(x.view(torch.uint8),y.view(torch.uint8)), 'finite':bool(torch.isfinite(x.float()).all() and torch.isfinite(y.float()).all()),
            'max_abs_error':float(e.abs().max()),'relative_rms':float(e.square().mean().sqrt()/y.float().double().square().mean().sqrt().clamp_min(1e-30)), 'elements':x.numel()}


def timed(fn):
    fn();torch.npu.synchronize();groups=[]
    for _ in range(3):
        t=time.perf_counter()
        for _ in range(10):value=fn()
        torch.npu.synchronize();groups.append((time.perf_counter()-t)*1000/10)
    return value,{'groups_ms':groups,'median_ms':statistics.median(groups)}


def chunk_table(cpu,expand,offset):
    m,w=cpu.shape
    whole=torch.full((m,expand*w),111.,dtype=cpu.dtype,device='npu')
    whole[:,offset*w:(offset+1)*w].copy_(cpu.to('npu'))
    return whole[:,offset*w:(offset+1)*w],whole


try:
    with torch.inference_mode():
        for name,kind,call in [('indexed_scale_shift_','mod',0),('indexed_gate','gate',0),('indexed_gate','gate',24),('indexed_gate','gate',49)]:
            data=torch.load(ROOT/'capture'/(name+'_%03d.pt'%call),weights_only=True)
            d=data['inputs'];x=d['x'].to('npu').contiguous();idx=d['indices'].to('npu')
            if kind=='mod':
                shift,shift_owner=chunk_table(d['shift'],2,0)
                scale,scale_owner=chunk_table(d['scale'],2,1)
                prepared=PreparedIndices(idx,shift.shape[0])
                old=lambda:ref.indexed_scale_shift_(x.clone(),shift,scale,idx)
                new=lambda:launch('mod',x,prepared,shift=shift,scale=scale)
                shapes={'shift_stride':list(shift.stride()),'scale_stride':list(scale.stride()),'shift_offset':shift.storage_offset(),'scale_offset':scale.storage_offset()}
            else:
                gate,gate_owner=chunk_table(d['gate'],6,5)
                other=d['other'].to('npu').contiguous();prepared=PreparedIndices(idx,gate.shape[0])
                old=lambda:ref.indexed_gate(x,gate,other,idx)
                new=lambda:launch('gate',x,prepared,gate=gate,branch=other)
                shapes={'gate_stride':list(gate.stride()),'gate_offset':gate.storage_offset()}
            before=x.cpu().contiguous()
            expected,et=timed(old);got,nt=timed(new)
            row={'name':name,'call':call,'shape':list(x.shape),**shapes,'vs_eager':bits(got,expected),'vs_actual_capture':bits(got,data['output']),
                 'input_unchanged':bits(x,before)['raw_bits_equal'],'eager_wrapper':et,'point_wrapper':nt}
            assert row['vs_eager']['raw_bits_equal'] and row['vs_actual_capture']['raw_bits_equal'] and row['input_unchanged']
            r['cases'].append(row);print(json.dumps(row),flush=True)
        # Exercise remainder-row loads, all six index values and true nonzero
        # chunk offsets without a contiguous-table shortcut.
        torch.manual_seed(620)
        for rows in (1,7,8,9,129):
            w=128;idx=(torch.arange(rows,device='npu')*5%6).long();prepared=PreparedIndices(idx,6)
            x=torch.randn(rows,w,dtype=torch.float16,device='npu');other=torch.randn_like(x)
            tables=torch.randn(6,6*w,dtype=x.dtype,device=x.device)
            shift,scale,gate=tables[:,0:w],tables[:,w:2*w],tables[:,5*w:6*w]
            a=launch('mod',x,prepared,shift=shift,scale=scale);b=ref.indexed_scale_shift_(x.clone(),shift,scale,idx)
            c=launch('gate',x,prepared,gate=gate,branch=other);d=ref.indexed_gate(x,gate,other,idx)
            torch.npu.synchronize()
            row={'rows':rows,'random_strided_mod':bits(a,b),'random_strided_gate':bits(c,d)}
            assert row['random_strided_mod']['raw_bits_equal'] and row['random_strided_gate']['raw_bits_equal'];r['cases'].append(row)
        original=torch.tensor([0,1,2,3,4,5,0],device='npu',dtype=torch.long)
        snap=PreparedIndices(original,6);original.zero_()
        r['guards']['snapshot_keeps_original_values']=torch.equal(snap.tensor[:7].cpu(),torch.tensor([0,1,2,3,4,5,0],dtype=torch.int32))
        assert r['guards']['snapshot_keeps_original_values']
        snap.tensor.add_(1)
        try:snap.checked(7,6,original.device)
        except AssertionError:r['guards']['prepared_mutation_rejected']=True
        else:raise RuntimeError('Prepared metadata mutation not rejected')
        snap2=PreparedIndices(original,6)
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            try:snap2.checked(7,6,original.device)
            except AssertionError:r['guards']['cross_stream_rejected']=True
            else:raise RuntimeError('Unvalidated cross-stream metadata accepted')
        try:PreparedIndices(torch.tensor([-1,0],device='npu'),6)
        except AssertionError:r['guards']['negative_index_rejected']=True
        else:raise RuntimeError('Negative index accepted')
        try:PreparedIndices(torch.tensor([6],device='npu'),6)
        except AssertionError:r['guards']['out_of_range_index_rejected']=True
        else:raise RuntimeError('Out-of-range index accepted')
        # The dense FP32 fallback has no stride ABI; its caller must reject
        # strided table inputs instead of passing incorrect shape semantics.
        x=torch.ones(2,64,dtype=torch.float16,device='npu');table=torch.ones(7,128,dtype=x.dtype,device=x.device)[:,64:]
        try:launch('mod',x,torch.tensor([0,6],device='npu'),shift=table,scale=table)
        except AssertionError:r['guards']['nonpoint_strided_fallback_rejected']=True
        else:raise RuntimeError('Stride-unsafe fallback accepted')
        r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:
    OUT.write_text(json.dumps(r,indent=2)+'\n');print('Evidence:',OUT,flush=True)
