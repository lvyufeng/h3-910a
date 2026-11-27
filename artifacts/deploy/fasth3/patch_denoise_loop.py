#!/usr/bin/env python3
"""Add the FastH3 PDD sampling path to the native MiniMax-H3 denoise loop."""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/data/h3fast/ref/vllm-omni/vllm_omni/diffusion/models/minimax_h3/denoise_loop.py")

SIG_OLD = """    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:"""
SIG_NEW = """    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
    pdd_plan: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:"""

DOC_OLD = """    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.
    \"\"\""""
DOC_NEW = """    conditional task, pass ``keyframe_cond_rows`` and/or ``audio_ref_rows`` to
    pin those rows across every step. The model's raw positive velocity is the
    update signal; MiniMax H3 only supports cfg-distilled checkpoints.

    ``pdd_plan`` selects the FastH3 PDD contract: the sigma schedules are
    replaced by the plan's per-modality node sigmas, each step's forward emits
    the block-fused rectified-flow direction of a widened final projection, and
    the state advances by ``x += (sigma_next - sigma_curr) * v`` (one Euler step
    over the block's shifted sigma interval).
    \"\"\""""

SIGMA_OLD = """    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")"""
SIGMA_NEW = """    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    if pdd_plan is not None:
        # PDD replaces the runtime schedule with the plan's node sigmas on each
        # modality's own (shifted) clock and prepares the fused final heads.
        model.prepare_pdd(pdd_plan)
        sigmas_video = [float(value) for value in pdd_plan.node_sigmas["video"]]
        sigmas_audio = [float(value) for value in pdd_plan.node_sigmas["audio"]]"""

FK_OLD = """            fk = positive.forward_kwargs(
                video_rows=model_video_rows,
                audio_rows=model_audio_rows,
                t_video=t_v,
                t_audio=t_a,
                imgvid_cond_timestep=imgvid_cond_t,
                audio_ref_cond_timestep=audio_ref_cond_t,
                video_target_timesteps=video_target_timesteps,
                audio_target_timesteps=audio_target_timesteps,
            )"""
FK_NEW = """            fk = positive.forward_kwargs(
                video_rows=model_video_rows,
                audio_rows=model_audio_rows,
                t_video=t_v,
                t_audio=t_a,
                imgvid_cond_timestep=imgvid_cond_t,
                audio_ref_cond_timestep=audio_ref_cond_t,
                video_target_timesteps=video_target_timesteps,
                audio_target_timesteps=audio_target_timesteps,
            )
            if pdd_plan is not None:
                fk["pdd_step"] = step"""

VIDEO_OLD = """            if video_edit is None:
                x0_video = minimax_h3_rf_v_to_x0(
                    video_rows[update],
                    mv_video_t,
                    torch.tensor(t_v, dtype=torch.float32, device=device),
                )
            else:
                x0_video = video_edit.x0(
                    model_video_rows[update],
                    mv_video_t,
                    t_v,
                )
            new_target = minimax_h3_euler_eta0_step(video_rows[update], x0_video, sigma_curr=s_v, sigma_next=s_v_next)"""
VIDEO_NEW = """            if pdd_plan is not None:
                # The fused head already is the block-integrated direction;
                # one Euler step over the block's shifted sigma interval.
                new_target = video_rows[update] + (s_v_next - s_v) * mv_video_t
            else:
                if video_edit is None:
                    x0_video = minimax_h3_rf_v_to_x0(
                        video_rows[update],
                        mv_video_t,
                        torch.tensor(t_v, dtype=torch.float32, device=device),
                    )
                else:
                    x0_video = video_edit.x0(
                        model_video_rows[update],
                        mv_video_t,
                        t_v,
                    )
                new_target = minimax_h3_euler_eta0_step(
                    video_rows[update], x0_video, sigma_curr=s_v, sigma_next=s_v_next
                )"""

AUDIO_OLD = """            if audio_edit is None:
                x0_audio = minimax_h3_rf_v_to_x0(
                    audio_rows[audio_update],
                    mv_audio_t,
                    torch.tensor(t_a, dtype=torch.float32, device=device),
                )
            else:
                x0_audio = audio_edit.x0(
                    model_audio_rows[audio_update],
                    mv_audio_t,
                    t_a,
                )
            new_audio = minimax_h3_euler_eta0_step(
                audio_rows[audio_update], x0_audio, sigma_curr=s_a, sigma_next=s_a_next
            )"""
AUDIO_NEW = """            if pdd_plan is not None:
                new_audio = audio_rows[audio_update] + (s_a_next - s_a) * mv_audio_t
            else:
                if audio_edit is None:
                    x0_audio = minimax_h3_rf_v_to_x0(
                        audio_rows[audio_update],
                        mv_audio_t,
                        torch.tensor(t_a, dtype=torch.float32, device=device),
                    )
                else:
                    x0_audio = audio_edit.x0(
                        model_audio_rows[audio_update],
                        mv_audio_t,
                        t_a,
                    )
                new_audio = minimax_h3_euler_eta0_step(
                    audio_rows[audio_update], x0_audio, sigma_curr=s_a, sigma_next=s_a_next
                )"""

IMPORT_OLD = "from typing import Any"  # placeholder probe
IMPORT_ANCHOR = "from collections.abc import Callable"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"anchor {label!r} matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text()
    if "pdd_plan" in source:
        raise SystemExit("already patched")
    out = replace_once(source, SIG_OLD, SIG_NEW, "signature")
    out = replace_once(out, DOC_OLD, DOC_NEW, "docstring")
    out = replace_once(out, SIGMA_OLD, SIGMA_NEW, "sigma override")
    out = replace_once(out, FK_OLD, FK_NEW, "forward kwargs")
    out = replace_once(out, VIDEO_OLD, VIDEO_NEW, "video update")
    out = replace_once(out, AUDIO_OLD, AUDIO_NEW, "audio update")
    TARGET.write_text(out)
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
