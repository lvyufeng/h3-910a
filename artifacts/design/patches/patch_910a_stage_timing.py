#!/usr/bin/env python3
"""Instrument the H3 pipeline's three top-level stages.

Per-step Python timers are not trustworthy on this device: the NPU runs
asynchronously, so a timer placed around the model call mostly measures how
much queued work the next stage had to drain. The `update` figure looked like
a fixed 14.7 s per step for exactly that reason, and did not move when the
checks inside it were removed.

Whole-stage wall clock over encode / denoise / decode is reliable, and it is
what showed that encode and decode cost 42 s and 157 s on a cold engine but
9 s and 43 s once warm -- neither depends on step count.

Enable with VLLM_OMNI_910A_PERF=1; output goes to stderr as [H3PERFP].

Usage: patch_910a_stage_timing.py /path/to/vllm_omni
"""
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/mseco/vllm-omni")
target = root / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
src = target.read_text(encoding="utf-8")

if "[H3PERFP]" in src:
    print("already patched")
    raise SystemExit(0)

edits = [
    (
        """        check_request_cancellation()
        context = self._prepare_request_inputs(
            request.prompts[0],
            request.sampling_params,
        )
        check_request_cancellation()
        denoise_kwargs = self._denoise_kwargs(context)""",
        """        check_request_cancellation()
        import os as _os_p, sys as _sys_p, time as _time_p
        _pf = _os_p.environ.get("VLLM_OMNI_910A_PERF", "0") != "0"
        _t0 = _time_p.perf_counter() if _pf else 0.0
        context = self._prepare_request_inputs(
            request.prompts[0],
            request.sampling_params,
        )
        check_request_cancellation()
        _t1 = _time_p.perf_counter() if _pf else 0.0
        if _pf:
            print("[H3PERFP] encode=%.2fs" % (_t1 - _t0), file=_sys_p.stderr, flush=True)
        denoise_kwargs = self._denoise_kwargs(context)""",
    ),
    (
        """            check_request_cancellation()
            if context["preencode_mp4"]:""",
        """            check_request_cancellation()
            _t2 = _time_p.perf_counter() if _pf else 0.0
            if _pf:
                print("[H3PERFP] denoise=%.2fs" % (_t2 - _t1), file=_sys_p.stderr, flush=True)
            if context["preencode_mp4"]:""",
    ),
    (
        """                video, audio = self.decode(
                    video_latent,
                    audio_latent,
                    height=context["height"],
                    width=context["width"],
                )""",
        """                video, audio = self.decode(
                    video_latent,
                    audio_latent,
                    height=context["height"],
                    width=context["width"],
                )
                if _pf:
                    print("[H3PERFP] decode=%.2fs" % (_time_p.perf_counter() - _t2),
                          file=_sys_p.stderr, flush=True)""",
    ),
]

for old, new in edits:
    if old not in src:
        print("anchor missing, aborting")
        raise SystemExit(1)
    src = src.replace(old, new, 1)

target.write_text(src, encoding="utf-8")
print("stage timing added to", target)
