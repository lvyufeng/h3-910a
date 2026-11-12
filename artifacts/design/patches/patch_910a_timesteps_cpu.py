#!/usr/bin/env python3
"""Keep MiniMax-H3's timestep bookkeeping off the NPU.

Two ops in this path are data dependent, and the 910A aclnn build cannot run
either of them:

* boolean-mask indexing (``positions[update_mask]``) lowers to ``NonZero``
  -> ``OPS function error: NonZero, error code is 500002``
* ``torch.unique(timesteps, return_inverse=True)`` runs on AICPU
  -> ``[Error]: The aicpu execution is abnormal`` / ``EE9999: Internal error``

All of it works on a 1-D vector of at most a few thousand rows, so the cheapest
correct fix is to do the bookkeeping on CPU and move the result back once. That
also avoids the device-side scatter (``timesteps[idx] = value``), which is
another dynamic-shape op.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/models/minimax_h3/denoise_loop.py")

OLD_UNIQUE = """        unique_timesteps, inverse_indices = torch.unique(timesteps, sorted=True, return_inverse=True)"""
NEW_UNIQUE = """        # 910A: torch.unique runs on AICPU and fails there ("The aicpu execution
        # is abnormal"), so do it on CPU; the tensor is a 1-D row vector.
        _unique_cpu, _inverse_cpu = torch.unique(
            timesteps.detach().cpu(), sorted=True, return_inverse=True
        )
        unique_timesteps = _unique_cpu.to(timesteps.device, dtype=timesteps.dtype)
        inverse_indices = _inverse_cpu.to(timesteps.device)"""

OLD_FILL = """        timesteps.fill_(float(t_video))"""
NEW_FILL = """        # 910A: build the packed timestep vector on CPU and copy it back once.
        # Device-side mask indexing needs NonZero and the scatter needs AICPU;
        # both are unavailable here, and the vector is tiny.
        _cpu_timesteps = torch.full(
            (self.seq_len,), float(t_video), dtype=torch.float32, device="cpu"
        )
        timesteps.fill_(float(t_video))"""

OLD_LOOP = """        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            # 910A: boolean-mask indexing lowers to NonZero (dynamic output
            # shape), which the aclnn build refuses. These are index vectors, so
            # select on CPU and push the result back.
            target_positions = positions.cpu()[update_mask.cpu()].to(positions.device)
            if target_timesteps is None:
                timesteps[target_positions] = current_timestep
            else:
                target_timesteps = target_timesteps.to(device=self.device, dtype=torch.float32).reshape(-1)
                if target_timesteps.shape[0] != target_positions.shape[0]:
                    raise ValueError(
                        f"{name} rows {target_timesteps.shape[0]} != target rows {target_positions.shape[0]}"
                    )
                timesteps[target_positions] = target_timesteps
            skip_positions = positions.cpu()[(~update_mask).cpu()].to(positions.device)
            timesteps[skip_positions] = condition_timestep"""

NEW_LOOP = """        for name, positions, update_mask, target_timesteps, current_timestep, condition_timestep in timestep_groups:
            cpu_positions = positions.detach().cpu()
            cpu_mask = update_mask.detach().cpu()
            target_positions = cpu_positions[cpu_mask]
            if target_timesteps is None:
                _cpu_timesteps[target_positions] = float(current_timestep)
            else:
                cpu_target = target_timesteps.detach().cpu().to(torch.float32).reshape(-1)
                if cpu_target.shape[0] != target_positions.shape[0]:
                    raise ValueError(
                        f"{name} rows {cpu_target.shape[0]} != target rows {target_positions.shape[0]}"
                    )
                _cpu_timesteps[target_positions] = cpu_target
            _cpu_timesteps[cpu_positions[~cpu_mask]] = float(condition_timestep)
        timesteps.copy_(_cpu_timesteps.to(timesteps.device, dtype=timesteps.dtype))"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "_cpu_timesteps" in text:
        print("already patched")
        return 0

    changed = []
    if OLD_UNIQUE in text:
        text = text.replace(OLD_UNIQUE, NEW_UNIQUE, 1)
        changed.append("unique")
    if OLD_FILL in text:
        text = text.replace(OLD_FILL, NEW_FILL, 1)
        changed.append("fill")
    if OLD_LOOP in text:
        text = text.replace(OLD_LOOP, NEW_LOOP, 1)
        changed.append("loop")

    if not changed:
        raise SystemExit("no anchors matched")
    TARGET.write_text(text, encoding="utf-8")
    print(f"patched: {', '.join(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())