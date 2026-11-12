#!/usr/bin/env python3
"""Skip fused RMSNorm on 910A.

``vllm_omni/diffusion/layers/norm.py`` calls ``torch_npu.npu_add_rms_norm``
(InplaceAddRmsNorm) for the residual path. That fused op has no kernel on the
910A generation:

    No supported Ops kernel and engine are found for [InplaceAddRmsNorm...]

The failure is asynchronous, so the surrounding ``try/except`` in ``forward()``
never sees it -- the error surfaces much later as a generic PTA failure. The
only reliable fix is to not issue the fused op at all on this hardware.

This patch makes ``forward_npu`` route to the existing ``forward_native``
implementation when ``VLLM_OMNI_910A_NO_FUSED_NORM`` is not ``0``.
Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/layers/norm.py")

OLD = """    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:"""

NEW = """    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        # 910A has no InplaceAddRmsNorm kernel, and the failure is async so it
        # cannot be caught around the call -- skip the fused path entirely.
        if os.environ.get("VLLM_OMNI_910A_NO_FUSED_NORM", "1") != "0":
            return self.forward_native(x, residual)

        if residual is not None:"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "VLLM_OMNI_910A_NO_FUSED_NORM" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    text = text.replace(OLD, NEW, 1)
    if "\nimport os\n" not in text and "^import os" not in text:
        # Ensure os is imported (module imports torch/F already).
        first_nl = text.index("\n")
        text = text[:first_nl + 1] + "\nimport os\n" + text[first_nl + 1:]
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())