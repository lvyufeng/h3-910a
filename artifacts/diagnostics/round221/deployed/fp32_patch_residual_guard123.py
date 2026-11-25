#!/usr/bin/env python3
"""Private precision guard for original FP32 H3 patch projections on 910A.

Keep the original FP32 parameter tensors. Reconstruct the projection through four
FP16-input/FP16-weight GEMMs with FP32 output/sum, using power-of-two scaled Half
residuals. No BF16, clipping, CPU computation, or shared installation changes.
This guard is not bit-exact FP32; same-input GPU leaf comparison is the acceptance
measurement. Do not treat leaf accuracy as full-model precision or speed success.
"""
from __future__ import annotations
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class ResidualFP32Projection123(UnquantizedLinearMethod):
    def __init__(self):
        super().__init__()
        self._weight_high_transposed = None
        self._weight_low_transposed = None
        self._cache_identity = None
        self.call_count = 0

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.device.type != 'npu' or x.dtype != torch.float32 or layer.weight.dtype != torch.float32:
            raise TypeError('Residual precision guard requires original FP32 x/weight on NPU')
        self.call_count += 1
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        scale = 4096.0
        identity = (layer.weight.data_ptr(), layer.weight._version, str(layer.weight.device))
        if self._cache_identity != identity:
            wh = layer.weight.half()
            wl = ((layer.weight - wh.float()) * scale).half()
            if not bool(torch.isfinite(wh).all() and torch.isfinite(wl).all()):
                raise FloatingPointError('The FP32 parameter high/low decomposition exceeds Half range')
            self._weight_high_transposed = wh.t().contiguous()
            self._weight_low_transposed = wl.t().contiguous()
            self._cache_identity = identity
        xh = flat.half()
        xl = ((flat - xh.float()) * scale).half()
        if not bool(torch.isfinite(xh).all() and torch.isfinite(xl).all()):
            raise FloatingPointError('The FP32 input high/low decomposition exceeds Half range')
        wh_t, wl_t = self._weight_high_transposed, self._weight_low_transposed
        output = torch.mm(xh, wh_t, out_dtype=torch.float32)
        output.add_(torch.mm(xh, wl_t, out_dtype=torch.float32), alpha=1.0 / scale)
        output.add_(torch.mm(xl, wh_t, out_dtype=torch.float32), alpha=1.0 / scale)
        output.add_(torch.mm(xl, wl_t, out_dtype=torch.float32), alpha=1.0 / (scale * scale))
        if bias is not None:
            output.add_(bias)
        return output.reshape(*shape[:-1], layer.weight.shape[0])


def install_patch_projection_guard123(model):
    affected = []
    for name in ['video_patch_proj', 'audio_patch_proj']:
        layer = getattr(model, name)
        if layer.weight.dtype != torch.float32:
            raise TypeError('Original patch projection parameters must remain FP32')
        layer.quant_method = ResidualFP32Projection123()
        affected.append(name)
    return affected
