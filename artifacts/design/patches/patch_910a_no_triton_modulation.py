#!/usr/bin/env python3
"""Skip the Triton modulation kernels on 910A.

``minimax_h3_transformer`` reaches ``indexed_modulation`` for its adaLN
modulation. Those helpers dispatch to ``@triton.jit`` kernels, and Triton cannot
build for this hardware at all:

    bishengir-compile: for the --target option:
    Cannot find option named 'Ascend910B'!

and the target list has no 910A entry either (it starts at Ascend910B1), so no
flag can make it work.

Each helper already carries a pure-PyTorch reference path for CPU tensors. This
patch makes NPU tensors take that same path when
``VLLM_OMNI_910A_EAGER_MODULATION`` is not ``0``.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/layers/indexed_modulation.py")

HELPER = '''

def _eager_modulation_910a() -> bool:
    """True when the Triton modulation kernels must be bypassed.

    Triton's Ascend backend targets Ascend910B1 and newer only; this hardware
    reports ``Ascend910B`` and has no matching bishengir target, so every
    ``@triton.jit`` kernel fails to compile.
    """
    import os

    return os.environ.get("VLLM_OMNI_910A_EAGER_MODULATION", "1") != "0"
'''

REPLACEMENTS = [
    ("    if x.is_cpu:\n", "    if x.is_cpu or _eager_modulation_910a():\n"),
    ("    if residual.is_cpu:\n", "    if residual.is_cpu or _eager_modulation_910a():\n"),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_eager_modulation_910a" in text:
        print("already patched")
        return 0

    anchor = "def _iter_row_chunks(rows: int):"
    if anchor not in text:
        raise SystemExit("anchor function missing")
    text = text.replace(anchor, HELPER.strip("\n") + "\n\n\n" + anchor, 1)

    total = 0
    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            total += count

    if not total:
        raise SystemExit("no guards matched")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {total} guard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())