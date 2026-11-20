"""R125 fused-pointwise candidate: SiLU-and-mul inside the DiT MLP.

The verified R124 graph computes the SwiGLU product in FP32 and then narrows it
for the second matrix:

    hidden, _ = self.fc1(x)
    gate, up = hidden.chunk(2, dim=-1)
    hidden = F.silu(gate.float()) * up.float()
    out, _ = self.fc2(hidden)

This replaces that sequence with the platform fused ``SiluAndMul`` operator, which
computes ``silu(gate) * up`` in the activation dtype in one pass. It removes two
FP32 elementwise kernels and one narrowing cast per block, and it changes the
rounding of the gate product from FP32 to FP16, so it is a lossy candidate that is
measured and reported, never automatically accepted.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def install_fused_mlp(module) -> dict:
    """Patch the replay MLP to use the fused SiluAndMul operator."""
    from vllm_omni.diffusion.layers.activation import SiluAndMul

    fused = SiluAndMul()
    original = module.MiniMaxH3MLP.forward

    def forward(self, x):
        hidden, _ = self.fc1(x)
        hidden = fused(hidden)
        out, _ = self.fc2(hidden)
        return out

    module.MiniMaxH3MLP.forward = forward
    return {
        'replaced': 'MiniMaxH3MLP.forward SwiGLU product',
        'original_math': 'F.silu(gate.float()) * up.float() in FP32, then narrowed for fc2',
        'variant_math': 'platform SiluAndMul in the activation dtype (FP16)',
        'original_is_restorable': callable(original),
        'operator': 'vllm_omni.diffusion.layers.activation.SiluAndMul',
    }