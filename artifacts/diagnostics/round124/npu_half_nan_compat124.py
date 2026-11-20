"""Same-semantics NPU NanToNum compatibility, scoped to native VAE decoding.

CANN lacks NanToNum here for both FP16 and FP32. Boolean selects preserve every
finite input bit while replacing only the original NaN/+Inf/-Inf entries with
native-dtype scalar replacements. No BF16, CPU model inference, matrix precision
changes, ad-hoc clipping, or nan policy changes. CPU work is scalar conversion
only. Other tensors/callers use the original ops.
"""
from __future__ import annotations
from contextlib import contextmanager
import threading

_STATE=threading.local()
_INSTALLED=False


def ieee_nonfinite_masks(input):
    """Classify NPU FP16/FP32 bit patterns without unreliable float predicates."""
    import torch
    import torch_npu
    assert input.device.type in ('npu','privateuseone') and input.dtype in (torch.float16,torch.float32)
    dense=torch_npu.npu_format_cast(input,2).contiguous()
    if input.dtype==torch.float16:
        bits=dense.view(torch.int16).to(torch.int32)
        magnitude=bits & 0x7fff
        nan_mask=magnitude>0x7c00
        inf_mask=magnitude==0x7c00
    else:
        bits=dense.view(torch.int32)
        exponent=bits & 0x7f800000
        mantissa=bits & 0x007fffff
        nan_mask=(exponent==0x7f800000) & (mantissa!=0)
        inf_mask=(exponent==0x7f800000) & (mantissa==0)
    return nan_mask,inf_mask & (bits>=0),inf_mask & (bits<0)


def install_half_nan_compat():
    global _INSTALLED
    if _INSTALLED:return
    import torch
    original=torch.nan_to_num
    original_method=torch.Tensor.nan_to_num
    original_inplace=torch.Tensor.nan_to_num_
    def functional(input,nan=0.0,posinf=None,neginf=None,*,out=None):
        active=getattr(_STATE,'active',False)
        if not (active and input.device.type in ('npu','privateuseone') and input.dtype in (torch.float16,torch.float32)):
            return original(input,nan=nan,posinf=posinf,neginf=neginf,out=out)
        bound=torch.finfo(input.dtype).max
        # Scalar replacements are rounded once to the native input dtype.
        # Transferring these tiny scalar tensors also preserves signed zero.
        def native_scalar(value,default):
            return torch.tensor(default if value is None else value,dtype=input.dtype,device='cpu').to(input.device)
        nan_mask,posinf_mask,neginf_mask=ieee_nonfinite_masks(input)
        result=torch.where(nan_mask,native_scalar(nan,0.0),input)
        result=torch.where(posinf_mask,native_scalar(posinf,bound),result)
        result=torch.where(neginf_mask,native_scalar(neginf,-bound),result)
        _STATE.calls=getattr(_STATE,'calls',0)+1
        if out is not None:out.copy_(result);return out
        return result
    def method(self,nan=0.0,posinf=None,neginf=None):
        if not getattr(_STATE,'active',False):return original_method(self,nan=nan,posinf=posinf,neginf=neginf)
        return functional(self,nan=nan,posinf=posinf,neginf=neginf)
    def inplace(self,nan=0.0,posinf=None,neginf=None):
        if not getattr(_STATE,'active',False):return original_inplace(self,nan=nan,posinf=posinf,neginf=neginf)
        result=functional(self,nan=nan,posinf=posinf,neginf=neginf)
        self.copy_(result);return self
    torch.nan_to_num=functional
    torch.Tensor.nan_to_num=method
    torch.Tensor.nan_to_num_=inplace
    _INSTALLED=True


@contextmanager
def half_nan_scope(enabled=True):
    if enabled:install_half_nan_compat()
    previous=getattr(_STATE,'active',False);before=getattr(_STATE,'calls',0)
    _STATE.active=bool(enabled)
    counter={'calls':0}
    try:yield counter
    finally:
        counter['calls']=getattr(_STATE,'calls',0)-before
        _STATE.active=previous
