"""Exact device-side power-of-two range selection for the R124 wide GEMMs.

For every finite Float32 peak p, the reference exponent is
max(0, ceil(log2(p/32752))). Equivalently it is the number of exact thresholds
32752*2**j strictly below p. Those thresholds, and powers of two, are exactly
representable in Float32. This avoids host scalar extraction and log2 without
changing rescale values, operand narrowing, FP32 GEMM output or bias order.

Not enabled until all real same-input leaf comparisons pass bitwise. Nonfinite
peaks are still rejected at the forward boundary before any sampler update.
"""
from __future__ import annotations
import math
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class DeviceRangeState126:
    def __init__(self):
        self.bad_peak_flags = []
        self.call_count = 0

    def begin(self):
        self.bad_peak_flags.clear()
        self.call_count = 0

    def end(self):
        if self.bad_peak_flags:
            count = int(torch.stack(self.bad_peak_flags).sum(dtype=torch.int32).cpu())
            if count:
                raise FloatingPointError(f'{count} nonfinite wide-GEMM input peaks')
        return {'range_selections_on_device': self.call_count, 'nonfinite_input_peaks': 0}


class ExactDeviceRangeLinear126(UnquantizedLinearMethod):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.tables = {}

    def prepare_tables(self, device):
        key = str(device)
        if key not in self.tables:
            # Integer ordering of positive IEEE Float32 values is exact even on
            # 910A versions with unreliable Float32 comparison predicates.
            thresholds = torch.tensor([32752.0*(2.0**j) for j in range(114)],
                                      dtype=torch.float32).view(torch.int32).to(device)
            scales = torch.tensor([2.0**j for j in range(115)], dtype=torch.float32, device=device)
            self.tables[key] = (thresholds, scales)
        return self.tables[key]

    def select(self, peak):
        thresholds, scales = self.prepare_tables(peak.device)
        positive_bits = peak.contiguous().view(torch.int32) & 0x7fffffff
        exponent = (positive_bits > thresholds).sum(dtype=torch.int32).long().reshape(1)
        return scales.index_select(0, exponent).reshape(())

    def apply(self, layer, x, bias=None):
        if x.device.type != 'npu' or layer.weight.dtype != torch.float16:
            raise TypeError('Exact-device trial requires original Half matrix weights on NPU')
        product = x.float()
        peak = product.abs().amax(dim=tuple(range(product.ndim)))
        bits = peak.contiguous().view(torch.int32)
        bad = ((bits & 0x7f800000) == 0x7f800000).to(torch.int32)
        self.state.bad_peak_flags.append(bad)
        self.state.call_count += 1
        scale = self.select(peak)
        half_input = (product / scale).to(torch.float16)
        result = torch.mm(half_input.reshape(-1, half_input.shape[-1]), layer.weight.t(),
                          out_dtype=torch.float32).reshape(*x.shape[:-1], layer.weight.shape[0])
        result = result * scale
        if bias is not None:
            result = result + bias.float()
        return result


def install_device_range(model):
    state = DeviceRangeState126()
    changed = []
    for name, module in model.named_modules():
        old = getattr(module, 'quant_method', None)
        if type(old).__name__ == '_MiniMaxH3HalfInputFP32OutputLinearMethod':
            new = ExactDeviceRangeLinear126(state)
            module.quant_method = new
            changed.append((name, module, old, new))
    assert len(changed) == 104, ('Unexpected original-model wide GEMM count', len(changed))
    return state, changed


def restore_device_range(changed):
    for name, module, old, new in changed:
        assert module.quant_method is new, name
        module.quant_method = old


def install_leaf_proof(model):
    """Return original outputs, but compare both algorithms on every actual input."""
    state, changed = install_device_range(model)
    rows = []
    for name, module, old, new in changed:
        candidate_apply = new.apply
        def apply(layer, x, bias=None, *, name=name, old=old, new=new, candidate_apply=candidate_apply):
            reference = old.apply(layer, x, bias)
            candidate = candidate_apply(layer, x, bias)
            assert reference.shape == candidate.shape and reference.dtype == candidate.dtype
            diff_words = int((reference.contiguous().view(torch.int32) !=
                              candidate.contiguous().view(torch.int32)).sum(dtype=torch.int32).cpu())
            peak = float(x.float().abs().amax(dim=tuple(range(x.ndim))).cpu())
            exponent = max(0, math.ceil(math.log2(peak/32752.0))) if peak>32752.0 else 0
            candidate_scale = float(new.select(torch.tensor(peak,dtype=torch.float32,device=x.device)).cpu())
            rows.append({'module':name,'shape':list(reference.shape),'elements':reference.numel(),
                         'different_words':diff_words,'original_peak':peak,
                         'reference_scale':2.0**exponent,'candidate_scale':candidate_scale,
                         'bit_identical':diff_words == 0,'scale_identical':candidate_scale == 2.0**exponent})
            return reference
        new.apply = apply
    return state, changed, rows
