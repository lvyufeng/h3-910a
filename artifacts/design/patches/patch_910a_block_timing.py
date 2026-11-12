#!/usr/bin/env python3
"""Log per-block timing inside the MiniMax-H3 DiT stack.

One denoise step measures ~14.5 minutes, which attention FLOPs cannot explain
(they are ~87s at peak across all four steps). The cost is therefore structural
and must be located rather than guessed. Logging every tenth block shows whether
the time is uniform across the stack (a per-layer cost, e.g. collectives or
offload traffic) or concentrated in one place (a pathological op).

Enabled with VLLM_OMNI_910A_PERF=1, so a normal run pays nothing.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

OLD = """        for block in self.blocks:
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, num_requests=num_requests)
        return self.final_norm(x)"""

NEW = """        import os as _os
        import time as _time

        _perf = _os.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
        _stack_t0 = _time.perf_counter() if _perf else 0.0
        for _bi, block in enumerate(self.blocks):
            _b0 = _time.perf_counter() if _perf else 0.0
            x = block(x, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, num_requests=num_requests)
            if _perf and (_bi % 10 == 0 or _bi + 1 == len(self.blocks)):
                _time.perf_counter()
                _el = _time.perf_counter() - _b0
                _tot = _time.perf_counter() - _stack_t0
                import sys as _sys

                print(
                    f"[H3PERF] block {_bi + 1}/{len(self.blocks)} "
                    f"block={_el:.3f}s stack={_tot:.3f}s",
                    file=_sys.stderr,
                    flush=True,
                )
        return self.final_norm(x)"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "[H3PERF]" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("patched block timing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())