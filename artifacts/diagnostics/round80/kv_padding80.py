#!/usr/bin/env python3
"""Zero-row-padding FP16 small-N projection stability control.

Original FP16 inputs and weight values,zero-only extra output rows;retain the
first128 columns. This checks a different GEMM tiling,not a model precision
pass. CPU nativeFP16 Linear is fixed;no BF16 path or global flags are changed.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_kv_pad01')
writer=EvidenceWriter(ROOT/'kv_padding.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[],'production_changed':False,'whole_model_precision_accepted':False,'whole_goal_complete':False}


def metric(a,b):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();e=a.double()-b.double()
    return {'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'finite':bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all()),'max_abs_error':float(e.abs().max()),'relative_RMS':float(e.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))}


try:
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(80081)
    x=(torch.randn(2043,5120)*.5).half();w=(torch.randn(128,5120)*.01).half()
    cpu=F.linear(x,w);xn=x.to('npu:0');wn=w.to('npu:0')
    for rows in(128,256,512,1024):
        padded=torch.zeros((rows,5120),dtype=torch.float16,device='npu:0');padded[:128].copy_(wn)
        with torch.inference_mode():
            baseline=F.linear(xn,padded)[...,:128].contiguous()
            repeats=[F.linear(xn,padded)[...,:128].contiguous()for _ in range(8)]
        row={'padded_output_rows':rows,'extra_rows_all_zero':bool(torch.count_nonzero(padded[128:])==0),
             'parameter_prefix_unchanged':bool(torch.equal(padded[:128].cpu(),w)),
             'same_input_CPU':metric(baseline,cpu),'repeat':[metric(item,baseline)for item in repeats]}
        r['cases'].append(row);writer.write(r)
    r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
