#!/usr/bin/env python3
"""Wire the FastH3 PDD plan into the native MiniMax-H3 pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/data/h3fast/ref/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py")

IMPORT_OLD = "    minimax_h3_time_shift_sigmas,\n"
IMPORT_NEW = (
    "    minimax_h3_time_shift_sigmas,\n"
    ")\n"
    "from .pdd import (\n"
    "    PDDModalitySchedule,\n"
    "    build_pdd_sampling_plan,\n"
)
# The import block ends with a closing paren; keep the module import after it.
IMPORT_ANCHOR = "from .fasth3 import FastH3WeightFusion, resolve_fasth3_fusion\n"
IMPORT_INSERT = (
    "from .fasth3 import FastH3WeightFusion, resolve_fasth3_fusion\n"
    "from .pdd import PDDModalitySchedule, build_pdd_sampling_plan\n"
)

SIGMA_OLD = """        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=video_shift,
            base_schedule=base_schedule,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=audio_shift,
            base_schedule=base_schedule,
        )"""
SIGMA_NEW = """        pdd_plan = None
        pdd_steps = int(getattr(getattr(self.transformer, "arch", None), "pdd_steps", 0) or 0)
        if pdd_steps > 0:
            if base_schedule is not None:
                raise ValueError("FastH3 PDD sampling does not take a base schedule override")
            if bool(getattr(self.od_config, "step_execution", False)):
                raise NotImplementedError(
                    "FastH3 PDD sampling currently requires request-mode execution "
                    "(step_execution=False): the fused block advance is not a scheduler step"
                )
            pdd_plan = build_pdd_sampling_plan(
                pdd_steps=pdd_steps,
                num_steps=num_steps,
                schedules={
                    "video": PDDModalitySchedule(video_shift),
                    "audio": PDDModalitySchedule(audio_shift),
                },
                device=self.device,
            )
            video_sigmas = [float(value) for value in pdd_plan.node_sigmas["video"]]
            audio_sigmas = [float(value) for value in pdd_plan.node_sigmas["audio"]]
        else:
            video_sigmas = minimax_h3_time_shift_sigmas(
                num_steps=num_steps,
                shift_scale=video_shift,
                base_schedule=base_schedule,
            )
            audio_sigmas = minimax_h3_time_shift_sigmas(
                num_steps=num_steps,
                shift_scale=audio_shift,
                base_schedule=base_schedule,
            )"""

RETURN_OLD = """            "sigmas_video": video_sigmas,
            "sigmas_audio": audio_sigmas,"""
RETURN_NEW = """            "sigmas_video": video_sigmas,
            "sigmas_audio": audio_sigmas,
            "pdd_plan": pdd_plan,"""

CALL_OLD = """                    video_edit=inputs["video_edit"],
                    audio_edit=inputs["audio_edit"],
                    on_step=lambda step, video, audio: progress.update(),
                )"""
CALL_NEW = """                    video_edit=inputs["video_edit"],
                    audio_edit=inputs["audio_edit"],
                    pdd_plan=inputs.get("pdd_plan"),
                    on_step=lambda step, video, audio: progress.update(),
                )"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"anchor {label!r} matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text()
    if "build_pdd_sampling_plan" in source:
        raise SystemExit("already patched")
    out = replace_once(source, IMPORT_ANCHOR, IMPORT_INSERT, "pdd import")
    out = replace_once(out, SIGMA_OLD, SIGMA_NEW, "sigma construction")
    out = replace_once(out, RETURN_OLD, RETURN_NEW, "inputs dict")
    out = replace_once(out, CALL_OLD, CALL_NEW, "loop call")
    TARGET.write_text(out)
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
