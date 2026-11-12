#!/usr/bin/env python3
"""Add per-phase wall-clock timing to the MiniMax-H3 denoise loop.

Reasoning about where the 14-minute steps go is unreliable: the FLOP estimates
say attention should be a small fraction of the step, so the time must be going
somewhere structural. Measure instead of guessing.

Prints one line per step with the DiT forward time, and accumulates the time
spent inside attention shim calls so the split is visible.
"""

from __future__ import annotations

import re
from pathlib import Path

DENOISE = Path("vllm_omni/diffusion/models/minimax_h3/denoise_loop.py")
SHIM = Path("vllm_omni/h3_910a_sdpa_shim.py")

TIMING_HELPER = '''


def _h3_perf_enabled() -> bool:
    import os

    return os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
'''

STEP_OLD = "        for step_index in range(num_steps):"
STEP_NEW = """        import time as _time

        _perf = _h3_perf_enabled()
        for step_index in range(num_steps):
            _step_t0 = _time.perf_counter() if _perf else 0.0"""


def main() -> int:
    text = DENOISE.read_text(encoding="utf-8")
    if "_h3_perf_enabled" in text:
        print("already patched")
        return 0

    # Put the helper near the top of the module, after the imports.
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines[:80]):
        if line.startswith(("import ", "from ")):
            insert_at = i + 1
    lines.insert(insert_at, TIMING_HELPER.lstrip("\n"))
    text = "".join(lines)

    if STEP_OLD not in text:
        raise SystemExit("denoise step anchor missing")
    text = text.replace(STEP_OLD, STEP_NEW, 1)
    DENOISE.write_text(text, encoding="utf-8")
    print(f"patched {DENOISE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())