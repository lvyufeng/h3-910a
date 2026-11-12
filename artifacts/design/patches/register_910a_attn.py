#!/usr/bin/env python3
"""Register the 910A attention backend as the TORCH_SDPA implementation."""

from pathlib import Path

REGISTRY = Path("vllm_omni/diffusion/attention/backends/registry.py")

BLOCK = """

# ---------------------------------------------------------------------------
# Ascend 910A: SDPA and the fused Ascend kernels are unavailable on this
# hardware generation, so route the TORCH_SDPA entry to the chunked
# matmul/softmax implementation. Non-NPU platforms keep the inherited SDPA
# path, because only forward_npu is overridden there.
# ---------------------------------------------------------------------------
register_diffusion_backend(
    DiffusionAttentionBackendEnum.TORCH_SDPA,
    "vllm_omni.diffusion.attention.backends.h3_910a_attn.H3910AAttentionBackend",
)
"""


def main() -> int:
    text = REGISTRY.read_text(encoding="utf-8")
    if "H3910AAttentionBackend" in text:
        print("already registered")
        return 0
    REGISTRY.write_text(text + BLOCK, encoding="utf-8")
    print(f"registered in {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())