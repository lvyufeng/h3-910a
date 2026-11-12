#!/usr/bin/env python3
"""Match the latent-embedder inputs to their projection weights' dtype.

``MiniMaxH3..._embed`` casts the selected rows to a hard-coded fp32 before
running the patch projections:

    x_rows = ...to(_FP32_DTYPE)
    video_embed, _ = self.video_patch_proj(x_rows)

The projections are declared "stay fp32" by design, but this deployment runs an
fp16 checkpoint, so under HSDP (which keeps these excluded modules at their
checkpoint dtype) the weight is Half while the input is Float:

    RuntimeError: expected dtype Float but got dtype Half
      ... minimax_h3_transformer.py line 1508 in _embed
      ... vllm/model_executor/layers/linear.py line 606 in forward

Cast the rows to whatever dtype the projection actually holds instead of to a
constant, which keeps the fp32 intent where the weights really are fp32 and
works unchanged for an fp16 checkpoint.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

OLD = """        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_global_pos).to(_FP32_DTYPE)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1])
        audio_rows = audio_rows.index_select(0, audio_global_pos).to(_FP32_DTYPE)"""

NEW = """        # 910A runs an fp16 checkpoint, so the projections may hold fp16 weights
        # even though this path was written for fp32. Follow the actual weight
        # dtype instead of assuming.
        _proj_dtype = self.video_patch_proj.weight.dtype
        _audio_proj_dtype = self.audio_patch_proj.weight.dtype
        x_rows = x.view(-1, x.shape[-1]).index_select(0, img_global_pos).to(_proj_dtype)
        video_embed, _ = self.video_patch_proj(x_rows)
        audio_rows = audio_x.view(-1, audio_x.shape[-1])
        audio_rows = audio_rows.index_select(0, audio_global_pos).to(_audio_proj_dtype)"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_proj_dtype" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())