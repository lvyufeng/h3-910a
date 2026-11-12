#!/usr/bin/env python3
"""Hard-disable torch.compile on 910A.

Ascend's graph back ends lower fp16 numerical stabilisation to ``nan_to_num``,
and 910A has no kernel for that op:

    No supported Ops kernel and engine are found for [NanToNum542],
    optype [NanToNum]

``TORCH_COMPILE_DISABLE`` is not sufficient because the video VAE and the
diffusion stack call ``torch.compile`` explicitly. Disabling dynamo globally
turns every such call into a no-op, which keeps the whole pipeline in eager mode.

The file must stay syntactically valid: vllm_omni imports it for side effects.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/h3_910a_sdpa_shim.py")

BLOCK = '''# --- hard-disable torch.compile ----------------------------------------------
# Ascend graph back ends lower fp16 stabilisation to nan_to_num, which has no
# kernel on 910A. TORCH_COMPILE_DISABLE is not enough because the VAE and the
# diffusion stack call torch.compile explicitly, so disable the back end itself.
try:
    import torch._dynamo as _dynamo

    _dynamo.config.disable = True
    _dynamo.config.suppress_errors = True
except Exception:
    pass

'''

MARKER = "# --- debug: trace where nan_to_num is requested"


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "hard-disable torch.compile" in text:
        print("already patched")
        return 0
    if MARKER not in text:
        raise SystemExit("anchor not found")
    text = text.replace(MARKER, BLOCK + MARKER, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched: dynamo disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())