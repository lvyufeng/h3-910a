"""R125 reference-only naive FP16 variant (NOT an accepted precision path).

Measures the speed upper bound of dropping the R123/R124 range protections:
plain FP16 matrix operands with an FP16 (not FP32) result, and plain Half patch
and time projections instead of the four-GEMM FP32-residual guards. R123 already
showed this shape saturates Half (finite 65504 values), so the result is expected
to be numerically unacceptable. It is executed only to quantify what the
precision contract costs; its outputs are recorded, never approved.
"""
from __future__ import annotations
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class PlainHalfLinearMethod(UnquantizedLinearMethod):
    def __init__(self):
        super().__init__()
        self._half_weight_t = None
        self._identity = None

    def apply(self, layer, x, bias=None):
        identity = (layer.weight.data_ptr(), layer.weight._version, str(layer.weight.device))
        if self._identity != identity:
            self._half_weight_t = layer.weight.half().t().contiguous()
            self._identity = identity
        flat = x.reshape(-1, x.shape[-1])
        out = torch.mm(flat.half(), self._half_weight_t)
        if bias is not None:
            out = out + bias.half()
        return out.reshape(*x.shape[:-1], layer.weight.shape[0]).float()


def install_naive(model):
    """Replace every guarded projection with the plain Half reference path."""
    touched = []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        quant_method = getattr(module, 'quant_method', None)
        if isinstance(quant_method, UnquantizedLinearMethod) and type(quant_method).__name__ == 'ResidualFP32Projection123':
            module.quant_method = PlainHalfLinearMethod()
            touched.append(name)
    if not touched:
        raise RuntimeError('no FP32 residual guard found to replace')
    return sorted(touched)