"""Measured-width FP32 projection padding for the 910A text-only experiment.

The real rank-local K/V has128 output channels. Its narrow FP32 dispatch gives
variable low bits even under deterministic_algorithms; padding to1024 output
channels and cropping gives repeatable same-input results in the measured case.
This changes FP32 reduction order, not checkpoint parameters. Default-off and
not installed in production. Cache is versioned and retains source storage.
"""
import torch
from torch.nn import functional as F

_CACHE = {}
_MAX_ENTRIES = 128


def supported(x, w):
    return (x.device.type in ('npu', 'privateuseone') and x.dtype == w.dtype == torch.bfloat16
            and x.shape == (1, 2043, 5120) and w.shape == (128, 5120)
            and x.device == w.device and not torch.is_grad_enabled())


def padded_weight(w):
    try:
        version = w._version
    except RuntimeError:
        return F.pad(w.float().contiguous(), (0, 0, 0, 896)).contiguous()
    key = (w.data_ptr(), version, tuple(w.shape), tuple(w.stride()), w.dtype, w.device)
    value = _CACHE.get(key)
    if value is None:
        if len(_CACHE) >= _MAX_ENTRIES:
            _CACHE.clear()
        value = (w, F.pad(w.float().contiguous(), (0, 0, 0, 896)).contiguous())
        _CACHE[key] = value
    return value[1]


def padded_linear(x, w):
    if not supported(x, w):
        raise ValueError('Unsupported shape/dtype/state for measured-width FP32 K/V padding')
    with torch.autocast(device_type=x.device.type, enabled=False):
        out = F.linear(x.float().contiguous(), padded_weight(w))
        return out[..., :128].contiguous().to(x.dtype)


def cache_stats():
    return {'entries': len(_CACHE), 'cached_fp32_bytes': sum(value[1].numel()*value[1].element_size() for value in _CACHE.values())}
