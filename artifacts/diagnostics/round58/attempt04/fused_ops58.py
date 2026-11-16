"""Default-off diagnostic wrappers for self-developed 910A fused vector kernels."""
import os
import torch
from torch.nn import functional as F
from fused_vector58 import launch

_WEIGHT_CACHE = {}


def _weight32(weight):
    try:
        version = weight._version
    except RuntimeError:
        return weight.float().contiguous()
    key = (id(weight), weight.data_ptr(), version, weight.dtype, weight.device)
    value = _WEIGHT_CACHE.get(key)
    if value is None:
        if len(_WEIGHT_CACHE) >= 256:
            _WEIGHT_CACHE.clear()
        # Keep the original weight alive to prevent data_ptr/id reuse.
        value = (weight, weight.float().contiguous())
        _WEIGHT_CACHE[key] = value
    return value[1]


def supported(x):
    return (x.device.type in ('npu', 'privateuseone')
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and x.ndim >= 2 and 64 <= x.shape[-1] <= 8192 and x.shape[-1] % 64 == 0
            and x.numel() > 0 and not (torch.is_grad_enabled() and x.requires_grad))


def eager_norm(x, weight, residual=None, eps=1e-6, qwen=True):
    with torch.autocast(device_type=x.device.type, enabled=False):
        merged = (x.float() + residual.float()).to(x.dtype) if residual is not None else x
        xf = merged.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        if qwen and x.dtype == torch.bfloat16:
            normalized = normalized.bfloat16().float()
        y = (normalized * weight.float()).to(x.dtype)
    return (y, merged) if residual is not None else y


def fused_norm(x, weight, residual=None, eps=1e-6, qwen=True):
    if not supported(x) or weight.numel() != x.shape[-1] or weight.device != x.device:
        raise ValueError('Unsupported shape/device for diagnostic fused norm')
    if residual is not None and (residual.shape != x.shape or residual.device != x.device or residual.dtype != x.dtype):
        raise ValueError('Residual contract mismatch')
    dtype = {torch.bfloat16: 'bf16', torch.float16: 'fp16', torch.float32: 'fp32'}[x.dtype]
    with torch.autocast(device_type=x.device.type, enabled=False):
        xf = x.float().contiguous()
        gamma = _weight32(weight)
        rf = residual.float().contiguous() if residual is not None else None
        y, updated = launch('addrms' if residual is not None else 'rms', xf, gamma, rf,
                             dtype=dtype, eps=eps, qwen=qwen)
        y = y.to(x.dtype)
        if residual is not None:
            return y, updated.to(x.dtype)
        return y


def norm(x, weight, residual=None, eps=1e-6, qwen=True):
    enabled = os.environ.get('VLLM_OMNI_910A_FUSED_VEC', '0').lower() in ('1', 'true', 'yes', 'on')
    if enabled and supported(x):
        return fused_norm(x, weight, residual, eps, qwen)
    return eager_norm(x, weight, residual, eps, qwen)


def fused_swiglu(gate, up):
    if not supported(gate) or up.shape != gate.shape or up.dtype != gate.dtype or up.device != gate.device:
        raise ValueError('Unsupported shape/device for diagnostic fused SwiGLU')
    dtype = {torch.bfloat16: 'bf16', torch.float16: 'fp16', torch.float32: 'fp32'}[gate.dtype]
    with torch.autocast(device_type=gate.device.type, enabled=False):
        y, _ = launch('swiglu', gate.float().contiguous(), up.float().contiguous(), dtype=dtype)
        return y.to(gate.dtype)


def swiglu(gate, up):
    enabled = os.environ.get('VLLM_OMNI_910A_FUSED_VEC', '0').lower() in ('1', 'true', 'yes', 'on')
    if enabled and supported(gate):
        return fused_swiglu(gate, up)
    return F.silu(gate) * up
