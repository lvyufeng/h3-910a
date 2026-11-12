#!/usr/bin/env python3
"""Make MiniMax-H3's packed SwiGLU eager on 910A.

``vllm_omni/platforms/npu/models/minimax_h3.py`` routes the Qwen3-VL text MLP
through ``torch_npu.npu_swiglu``. On the 910A generation that aclnn op refuses to
run:

    EZ1009: Failed to execute operator SwiGlu_0.
    Reason: 1.SoC version ascend910 verification failed.
            This SoC is not configured through the AddConfig API of the OpDef class.

The kernel binary simply is not built for ascend910, so no environment tweak can
enable it. ``silu(x1) * x2`` is exactly what the fused op computes.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/platforms/npu/models/minimax_h3.py")

OLD = '''def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply fused SwiGLU to a packed gate/up projection tensor."""
    return torch_npu.npu_swiglu(gate_up, dim=-1)'''

NEW = '''def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to a packed gate/up projection tensor.

    910A ships no aclnnSwiGlu kernel -- the binary is gated on SoC version
    (EZ1009 "SoC version ascend910 verification failed"), so the fused call
    cannot run here at all. The eager form computes the same thing.
    """
    import os

    if os.environ.get("VLLM_OMNI_910A_NO_FUSED_ACT", "1") != "0":
        half = gate_up.shape[-1] // 2
        return torch.nn.functional.silu(gate_up[..., :half]) * gate_up[..., half:]
    return torch_npu.npu_swiglu(gate_up, dim=-1)'''


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