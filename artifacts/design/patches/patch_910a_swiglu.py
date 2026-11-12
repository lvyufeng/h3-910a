#!/usr/bin/env python3
"""Fall back to eager SwiGLU when the fused Ascend op is missing.

``torch_npu.npu_swiglu`` (aclnnSwiGlu) does not exist on the 910A generation --
it fails for both fp16 and bf16 with

    npu_swiglu: ... StructKernelNpuOpApi_23.cpp:168

while ``F.silu(x1) * x2`` runs fine in eager. vllm-ascend's ``AscendSiluAndMul``
calls the fused op unconditionally, and its failure surfaces later as an
unrelated ``aclrtAllocatorGetByStream`` from an asynchronous all_reduce.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("/home/mseco/vllm910a/lib/python3.12/site-packages/vllm_ascend/ops/activation.py")

OLD = '''class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        out = torch_npu.npu_swiglu(x)
        return out
'''

NEW = '''def _eager_swiglu(x: torch.Tensor) -> torch.Tensor:
    """silu(x1) * x2 over the last dimension (matches npu_swiglu semantics)."""
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    return torch.nn.functional.silu(gate) * up


class AscendSiluAndMul(SiluAndMul):
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # 910A has no aclnnSwiGlu kernel; eager is the only working path.
        if os.environ.get("VLLM_ASCEND_910A_NO_FUSED_ACT", "1") != "0":
            return _eager_swiglu(x)
        out = torch_npu.npu_swiglu(x)
        return out
'''

OLD2 = '''        x = torch.cat([gate, up], dim=-1)
        out = torch_npu.npu_swiglu(x)
        return out
'''

NEW2 = '''        x = torch.cat([gate, up], dim=-1)
        if os.environ.get("VLLM_ASCEND_910A_NO_FUSED_ACT", "1") != "0":
            return _eager_swiglu(x)
        out = torch_npu.npu_swiglu(x)
        return out
'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_eager_swiglu" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor for AscendSiluAndMul missing")

    text = text.replace(OLD, NEW, 1)
    if OLD2 in text:
        text = text.replace(OLD2, NEW2, 1)

    if "\nimport os\n" not in text:
        first = text.index("\n")
        text = text[: first + 1] + "\nimport os\n" + text[first + 1 :]

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())