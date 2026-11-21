# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Custom activation functions for diffusion models."""

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.platforms import current_omni_platform


class SiluAndMul(CustomOp):
    """SwiGLU activation dispatched by the current vLLM-Omni platform."""

    def __init__(self) -> None:
        super().__init__()
        if current_omni_platform.is_cuda() or current_omni_platform.is_rocm() or current_omni_platform.is_musa():
            self.op = torch.ops._C.silu_and_mul

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu":
            return self.forward_native(x)
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        # 910A ships no aclnnSwiGlu kernel (SoC version gate, EZ1009), so the
        # fused dispatch cannot run; the eager form is the same math.
        import os

        if os.environ.get("VLLM_OMNI_910A_NO_FUSED_ACT", "1") != "0":
            return self.forward_native(x)

        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)
