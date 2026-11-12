#!/usr/bin/env python3
"""Do MiniMax-H3's mask indexing on CPU so 910A never sees NonZero.

``fill_timesteps`` selects rows with boolean-mask indexing:

    target_positions = positions[update_mask]
    timesteps[target_positions] = ...
    timesteps[positions[~update_mask]] = ...

Under the hood that is ``nonzero``, whose output shape is data dependent. The
910A aclnn build fails it with

    OPS function error: NonZero, error code is 500002

These tensors hold row indices (a few thousand int64 at most), so computing the
selection on CPU costs microseconds and sidesteps the dynamic-shape op entirely.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/denoise_loop.py")

OLD = """        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            target_positions = positions[update_mask]"""

NEW = """        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            # 910A: boolean-mask indexing lowers to NonZero (dynamic output
            # shape), which the aclnn build refuses. These are index vectors, so
            # select on CPU and push the result back.
            target_positions = positions.cpu()[update_mask.cpu()].to(positions.device)"""

OLD2 = """            timesteps[positions[~update_mask]] = condition_timestep"""

NEW2 = """            skip_positions = positions.cpu()[(~update_mask).cpu()].to(positions.device)
            timesteps[skip_positions] = condition_timestep"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "lowers to NonZero" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor 1 missing")
    text = text.replace(OLD, NEW, 1)
    if OLD2 in text:
        text = text.replace(OLD2, NEW2, 1)
    else:
        print("warning: skip-mask anchor missing")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())