"""Exact-count Half range audit with a cheap common path (R377).

Counts are identical to equivalent_safety126.half_safety_counts(fast=True):
  bad      = #{w : (bits & 0x7fff) >= 0x7c00}   (NaN or +/-Inf)
  boundary = #{w : (bits & 0x7fff) == 0x7bff}   (|x| == 65504)

Common path: a single |x| max reduction. If peak is finite and < 65504 then no
element can be NaN, +/-Inf or the boundary value, so both counts are exactly 0.
Any other outcome falls back to the original word-level counting path, so the
numbers, the strict-mode exceptions and the strict host reads are unchanged.
"""
from __future__ import annotations
import torch
import torch_npu

BOUNDARY = 65504.0


def _exact_counts(value):
    dense = torch_npu.npu_format_cast(value, 2).contiguous()
    magnitude = dense.view(torch.int16).to(torch.int32) & 0x7fff
    return (int((magnitude >= 0x7c00).sum(dtype=torch.int32)),
            int((magnitude == 0x7bff).sum(dtype=torch.int32)))


def half_safety_counts(value, fast=False):
    assert value.dtype == torch.float16 and value.device.type == "npu"
    if not fast:
        from npu_half_nan_compat124 import ieee_nonfinite_masks
        nan, posinf, neginf = ieee_nonfinite_masks(value)
        return ((nan | posinf | neginf).sum(dtype=torch.int32),
                (value.abs() == 65504).sum(dtype=torch.int32))
    peak = float(value.abs().max())
    if peak == peak and peak < BOUNDARY:      # finite and below the boundary
        return (0, 0)
    return _exact_counts(value)
