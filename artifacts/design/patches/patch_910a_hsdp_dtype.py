#!/usr/bin/env python3
"""Make HSDP's all-gather dtype follow the configured model dtype.

``HSDPConfig.param_dtype`` defaults to ``torch.bfloat16``. On this deployment the
MiniMax-H3 checkpoint was converted to fp16 (the fused 910A attention path only
handles fp16), so FSDP2 all-gathered parameters as bf16 while the surrounding
graph still expected Half, and the first forward died with:

    expected dtype Float but got dtype Half

Follow the dtype the launcher actually selected instead of hard-coding bf16.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/distributed/hsdp.py")

OLD = "    param_dtype: torch.dtype = torch.bfloat16"
NEW = """    param_dtype: torch.dtype = _default_hsdp_param_dtype()"""

HELPER = '''

def _default_hsdp_param_dtype() -> torch.dtype:
    """Parameter dtype for FSDP2 all-gather.

    Defaults to bf16 (upstream behaviour) but follows
    ``VLLM_OMNI_HSDP_PARAM_DTYPE`` when set, so a converted fp16 checkpoint does
    not hit "expected dtype Float but got dtype Half".
    """
    import os

    name = os.environ.get("VLLM_OMNI_HSDP_PARAM_DTYPE", "").strip().lower()
    mapping = {
        "half": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)

'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_default_hsdp_param_dtype" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    text = text.replace(OLD, NEW, 1)

    anchor = "class HSDPInferenceConfig"
    if anchor not in text:
        raise SystemExit("class anchor not found")
    idx = text.index(anchor)
    # Insert the helper right before the class definition.
    line_start = text.rfind("\n", 0, idx) + 1
    text = text[:line_start] + HELPER.strip("\n") + "\n\n\n" + text[line_start:]

    TARGET.write_text(text, encoding="utf-8")
    print("patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())