#!/usr/bin/env python3
"""Route vLLM-Omni's diffusion SwiGLU to eager on 910A.

``vllm_omni/diffusion/layers/activation.py`` dispatches ``SiluAndMul.forward_npu``
to ``torch_npu.npu_swiglu``. The aclnn kernel for that op is gated on SoC version
and refuses ascend910:

    EZ1009: Failed to execute operator SwiGlu_0.
    Reason: 1.SoC version ascend910 verification failed.

The class already carries an eager ``forward_native``; use it on this hardware.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/layers/activation.py")

OLD = '''    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)'''

NEW = '''    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        # 910A ships no aclnnSwiGlu kernel (SoC version gate, EZ1009), so the
        # fused dispatch cannot run; the eager form is the same math.
        import os

        if os.environ.get("VLLM_OMNI_910A_NO_FUSED_ACT", "1") != "0":
            return self.forward_native(x)

        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "EZ1009" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())