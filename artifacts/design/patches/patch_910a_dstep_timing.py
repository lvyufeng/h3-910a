#!/usr/bin/env python3
"""Time the three phases of one H3 denoise step.

``[H3PERF] phase`` showed the DiT forward is only ~7.5s of a ~102s step for a
10-second video, so the waste is outside the model. The loop body splits
naturally into row preparation, the model call, and the scheduler update; timing
those separately says which one to attack.

Emits, gated on ``VLLM_OMNI_910A_PERF``, to stderr:

    [H3PERF] dstep 2 prep=0.4s model=7.5s update=94.1s

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/denoise_loop.py")

PREP_ANCHOR = """            with torch.inference_mode():
                v_video, v_audio = model(**fk)"""

PREP_BLOCK = """            import os as _os_d
            import sys as _sys_d
            import time as _time_d

            _perf_d = _os_d.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
            _dprep = 0.0
            if _perf_d:
                _dt_all = _time_d.perf_counter() - _dstep_t0
            with torch.inference_mode():
                _dm0 = _time_d.perf_counter() if _perf_d else 0.0
                v_video, v_audio = model(**fk)
                _dmodel = _time_d.perf_counter() - _dm0 if _perf_d else 0.0"""

UPDATE_ANCHOR = """            if on_step is not None:
                on_step(step, video_rows, audio_rows)"""

UPDATE_BLOCK = """            if _perf_d:
                _dend = _time_d.perf_counter()
                print(
                    f"[H3PERF] dstep {step + 1} prep={_dt_all:.2f}s"
                    f" model={_dmodel:.2f}s"
                    f" update={_dend - _dm0 - _dmodel:.2f}s",
                    file=_sys_d.stderr,
                    flush=True,
                )
            if on_step is not None:
                on_step(step, video_rows, audio_rows)"""

T0_ANCHOR = """        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()"""

T0_BLOCK = """        import time as _time_dt

        _dstep_t0 = _time_dt.perf_counter()
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "[H3PERF] dstep" in text:
        print("already patched")
        return 0
    for anchor, name in (
        (T0_ANCHOR, "step start"),
        (PREP_ANCHOR, "model call"),
        (UPDATE_ANCHOR, "on_step"),
    ):
        if anchor not in text:
            raise SystemExit(f"{name} anchor not found")
    text = text.replace(T0_ANCHOR, T0_BLOCK, 1)
    text = text.replace(PREP_ANCHOR, PREP_BLOCK, 1)
    text = text.replace(UPDATE_ANCHOR, UPDATE_BLOCK, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("patched: denoise step phase timing added")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())