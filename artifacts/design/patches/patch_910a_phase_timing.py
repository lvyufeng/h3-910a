#!/usr/bin/env python3
"""Add phase-level timing to the H3 DiT forward.

The existing ``[H3PERF] dit`` log only covers the 50-block loop, and that loop
turned out to be a small slice of a denoise step (~2.2s of ~21s under fp16).
This prints how the remaining time divides between preprocessing and the
post-block tail, so the next optimization targets the real cost.

Emits, gated on ``VLLM_OMNI_910A_PERF``, to stderr:

    [H3PERF] phase pre=1.234s blocks=2.226s post=0.456s total=3.916s

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

# Forward entry: start the clock before any preprocessing runs.
ENTRY_ANCHOR = '''        # Strict keyword contract: refuse any kwarg forward does not consume.
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)'''

ENTRY_BLOCK = '''        # Strict keyword contract: refuse any kwarg forward does not consume.
        import time as _pt

        _t_entry = _pt.perf_counter()
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)'''

# Right before the 50-block loop.
LOOP_ANCHOR = '''        _perf2 = _os2.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
        _st_t0 = _time2.perf_counter() if _perf2 else 0.0'''

LOOP_BLOCK = '''        _perf2 = _os2.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
        _pre_s = _pt.perf_counter() - _t_entry
        _st_t0 = _time2.perf_counter() if _perf2 else 0.0'''

# After the tail is done, just before the output-row selection.
TAIL_ANCHOR = '''        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.'''

TAIL_BLOCK = '''        if _perf2:
            _s_now = _time2.perf_counter()
            _blocks_s = _s_now - _st_t0
            print(
                f"[H3PERF] phase pre={_pre_s:.3f}s blocks={_blocks_s:.3f}s"
                f" post={_s_now - _t_entry - _pre_s - _blocks_s:.3f}s"
                f" total={_s_now - _t_entry:.3f}s",
                file=_sys2.stderr,
                flush=True,
            )

        # Select target and condition rows at inference-output positions, then
        # zero the condition rows.'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "[H3PERF] phase" in text:
        print("already patched")
        return 0
    for anchor, name in (
        (ENTRY_ANCHOR, "entry"),
        (LOOP_ANCHOR, "block loop"),
        (TAIL_ANCHOR, "tail"),
    ):
        if anchor not in text:
            raise SystemExit(f"{name} anchor not found")
    text = text.replace(ENTRY_ANCHOR, ENTRY_BLOCK, 1)
    text = text.replace(LOOP_ANCHOR, LOOP_BLOCK, 1)
    text = text.replace(TAIL_ANCHOR, TAIL_BLOCK, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched: phase timing added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())