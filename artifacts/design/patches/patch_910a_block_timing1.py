#!/usr/bin/env python3
"""Log per-block timing inside the MiniMax-H3 DiT stack.

The measured 14min29s for one denoise step cannot be explained by attention
FLOPs (they account for ~87s at peak across all four steps), so the cost is
structural and has to be located, not guessed. Logging every tenth block shows
whether time is uniform across the stack (per-layer cost) or concentrated
(one pathological layer), which points at very different optimisations.

Enabled with VLLM_OMNI_910A_PERF=1 so a normal run pays nothing.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

OLD = "        for block in self.blocks:"
NEW = """        import os as _os
        import time as _time

        _perf = _os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
        for _block_idx, block in enumerate(self.blocks):
            _bt0 = _time.perf_counter() if _perf else 0.0"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_block_idx, block in enumerate(self.blocks)" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    text = text.replace(OLD, NEW, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched block loop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())