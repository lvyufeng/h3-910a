#!/usr/bin/env python3
"""Check native FP16 Linear with nonzero-storage-offset slices against zero-offset clones.

Same-input one-device operator probe;does not replay TP8 or alter production.
The original TP8 log warns that nonzero storage offsets are untrustworthy.
Torch contiguous() is insufficient when a contiguous slice already has a
nonzero offset;clone() materializes a real zero-offset operand.
"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from torch.nn import functional as F
from evidence_io80 import EvidenceWriter

ROOT=Path('/tmp/h3_round80_offset01')
writer=EvidenceWriter(ROOT/'offset.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[],'production_changed':False,'whole_goal_complete':False}


def metric(a,b):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();e=a.double()-b.double()
    return {'raw_bits_equal':torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'finite':bool(torch.isfinite(a.float()).all()and torch.isfinite(b.float()).all()),'max_abs_error':float(e.abs().max()),'relative_RMS':float(e.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30))}


try:
    torch.set_num_threads(8);torch.npu.set_device(0);torch.manual_seed(80081)
    for name,M,K,N,prefix in(('kv_small',2043,5120,128,1024),('mlp_up',2043,5120,3456,3456)):
        x=(torch.randn(M,K)*.5).half();base=(torch.randn(N+prefix,K)*.01).half()
        source=base[prefix:];cpu=F.linear(x,source)
        xn=x.to('npu:0');bn=base.to('npu:0');view=bn[prefix:];clone=view.clone()
        with torch.inference_mode():
            a=F.linear(xn,view);b=F.linear(xn,clone);c=F.linear(xn,view);d=F.linear(xn,clone)
        r['cases'].append({'name':name,'shape':{'M':M,'K':K,'N':N},'view_storage_offset':view.storage_offset(),'view_is_contiguous':view.is_contiguous(),
                           'contiguous_still_offset':view.contiguous().storage_offset(),'clone_storage_offset':clone.storage_offset(),
                           'view_vs_clone':metric(a,b),'view_repeat':metric(a,c),'clone_repeat':metric(b,d),'view_vs_CPU':metric(a,cpu),'clone_vs_CPU':metric(b,cpu)})
        writer.write(r)
    r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
