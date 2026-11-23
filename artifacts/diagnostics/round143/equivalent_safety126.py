"""R126 equivalent Half safety counts with fewer read-only integer operations.

For every FP16 word, magnitude >= 0x7c00 iff it is NaN or either infinity;
magnitude == 0x7bff iff abs(value) == 65504. This combines the ORIGINAL tests,
without removing either check, converting model values, clipping, or sanitizing.
Strict mode still reads counts before the next module, as in R124.
"""
from __future__ import annotations
import torch
import torch_npu
from matrix_safety126 import MatrixSafetyAudit126
from npu_half_nan_compat124 import ieee_nonfinite_masks


def half_safety_counts(value,fast=False):
    assert value.dtype==torch.float16 and value.device.type=='npu'
    if not fast:
        nan,posinf,neginf=ieee_nonfinite_masks(value)
        return ((nan|posinf|neginf).sum(dtype=torch.int32),
                (value.abs()==65504).sum(dtype=torch.int32))
    dense=torch_npu.npu_format_cast(value,2).contiguous()
    magnitude=dense.view(torch.int16).to(torch.int32) & 0x7fff
    return ((magnitude>=0x7c00).sum(dtype=torch.int32),
            (magnitude==0x7bff).sum(dtype=torch.int32))


class EquivalentMatrixSafetyAudit126(MatrixSafetyAudit126):
    def __init__(self,model,mode='strict'):
        self.fast=False
        self.profile=False
        self.event_pairs=[(torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)) for _ in range(160)]
        self.profile_names=[]
        super().__init__(model,mode)

    def _hook(self,name):
        def hook(module,inputs,result):
            value=result[0] if isinstance(result,tuple) else result
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float16:return
            index=self.calls
            self.calls+=1
            if self.profile:self.event_pairs[index][0].record()
            bad,boundary=half_safety_counts(value,self.fast)
            if self.mode=='strict':
                nbad,nboundary=int(bad),int(boundary)
                if nbad or nboundary:
                    raise FloatingPointError(f'Half matrix range failure at {name}: {nbad}, {nboundary}')
            else:
                self.counts.append(torch.stack((bad,boundary)));self.names.append(name)
            if self.profile:
                self.event_pairs[index][1].record();self.profile_names.append(name)
        return hook

    def begin(self):
        super().begin();self.profile_names.clear()

    def profile_receipt(self):
        assert self.calls==155 and len(self.profile_names)==155
        milliseconds=[a.elapsed_time(b) for a,b in self.event_pairs[:155]]
        return {'fast_integer_equivalent':self.fast,'calls':155,'device_span_ms':sum(milliseconds),
                'per_matrix':[{'module':n,'device_span_ms':t} for n,t in zip(self.profile_names,milliseconds)],
                'same_two_range_checks_and_exact_counts':True,
                'strict_host_reads_not_removed':self.mode=='strict',
                'model_values_modified':False,'profile_instrumented_not_clean':True}
