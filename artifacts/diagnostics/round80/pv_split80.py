#!/usr/bin/env python3
"""Same-input split-FP16 probability PV control,not a CPU reference change.

On this910A,arbitrary-FP32 probability matmul has a measured2e-4 gap even
with allow_hf32=false. Both operands are FP16-exact in each compensated GEMM;
combine FP32 outputs after an exact power-of-two residual scale. No bias,
clamps,quantized weights or global math flags are changed.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_pv_split01')
writer=EvidenceWriter(ROOT/'pv_split.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':{},'CPU_reference_changed':False,'production_changed':False,'whole_model_precision_accepted':False,'whole_goal_complete':False}


def metric(a,b):
    a=a.detach().cpu().double();b=b.detach().cpu().double();e=a-b
    return {'finite':bool(torch.isfinite(a).all()and torch.isfinite(b).all()),'max_abs_error':float(e.abs().max()),'RMSE':float(e.square().mean().sqrt()),'relative_RMS':float(e.square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-30))}


try:
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(801235)
    r['matmul_allow_hf32_observed']=bool(torch.npu.matmul.allow_hf32)
    p=torch.softmax(torch.randn(1,16,256,256)*2,dim=-1);v=torch.randn(1,16,256,72,dtype=torch.float16)
    with torch.inference_mode():
        cpu=p@v.float();pn=p.to('npu:0');vn=v.to('npu:0').float()
        native=pn@vn;r['cases']['native_FP32']=metric(native,cpu)
        for scale in(1.,4096.,65536.):
            high=pn.to(torch.float16).float()
            low=((pn-high)*scale).to(torch.float16).float()
            a=high@vn; b=low@vn; combined=a+b/scale
            r['cases']['split_scale_%d'%scale]={'same_input_CPU_FP32':metric(combined,cpu),'FP32_probability_reconstruction':metric(high+low/scale,p),
                                              'operand_high_is_FP16_exact':bool(torch.equal(high,high.half().float())),'operand_low_is_FP16_exact':bool(torch.equal(low,low.half().float()))}
            assert r['cases']['split_scale_%d'%scale]['same_input_CPU_FP32']['finite']
        r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
