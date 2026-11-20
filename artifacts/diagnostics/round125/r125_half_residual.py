"""R125 FP16-residual-stream variant of the verified replay transformer.

The verified R124 graph keeps the block residual stream in FP32: its
``_modulate_gate`` explicitly widens both operands before the gated add, because a
real block-0 result exceeded the Half range. This variant narrows that state to
Half instead, which is exactly the "FP16 residual stream" acceleration the user
approved for measurement.

Scope of this variant, stated honestly: it replaces the module-level
``_modulate_gate`` used by the 50 main DiT blocks. The two-layer token refiner
keeps its own FP32 accumulation because that path is defined inside a class
method rather than a module function, so the refiner is unchanged. The forward
is therefore a lower bound on the deviation of a complete FP16-stream rewrite.
"""
from __future__ import annotations
import torch


def half_modulate_gate(x, gate, other, indices, *, dtype):
    """Gated residual accumulation in the incoming Half dtype."""
    del dtype
    return x + gate.index_select(0, indices) * other


def install_half_residual(module) -> dict:
    original = module._modulate_gate
    module._modulate_gate = half_modulate_gate
    return {
        'replaced_module_function': '_modulate_gate',
        'original_widened_to': 'float32',
        'variant_keeps': 'incoming dtype (float16)',
        'uncovered_path': 'MiniMaxH3TokenRefinerBlock.forward still accumulates in FP32',
        'original_is_restorable': callable(original),
    }