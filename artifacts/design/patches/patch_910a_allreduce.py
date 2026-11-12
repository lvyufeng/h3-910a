#!/usr/bin/env python3
"""Make diffusion-group collectives safe on 910A.

``minimax_h3/encoder.py`` masks out-of-partition token ids and then all-reduces
the embedding output across the tensor-parallel group. On 910A that collective
fails with

    HcclAllreduce, error code is 1 / ERR01100 OPS call acl api failed
    [Error]: parameter error

Two things upset HCCL here: the tensor arrives non-contiguous (masked in-place
assignment), and it is bfloat16, which this hardware generation does not handle
for every collective. This patch makes the input contiguous and, on NPU,
downcasts bf16 to fp16 for the collective and back afterwards.

Idempotent.
"""

from __future__ import annotations

from pathlib import Path

TARGET = Path("vllm_omni/diffusion/distributed/group_coordinator.py")

OLD = """        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_reduce(input_, op=op, group=self.device_group)
        return input_
"""

NEW = """        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # 910A: HCCL rejects non-contiguous inputs ("parameter error") and does
        # not support bfloat16 for every collective, so normalise both here.
        input_ = input_.contiguous()
        if input_.dtype is torch.bfloat16:
            promoted = input_.to(torch.float16)
            torch.distributed.all_reduce(promoted, op=op, group=self.device_group)
            input_.copy_(promoted.to(input_.dtype))
            return input_
        torch.distributed.all_reduce(input_, op=op, group=self.device_group)
        return input_
"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    if "910A: HCCL rejects non-contiguous inputs" in text:
        print("already patched")
        return 0
    if OLD not in text:
        raise SystemExit("anchor not found")
    TARGET.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())