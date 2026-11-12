#!/usr/bin/env python3
"""Check which HCCL collectives actually work on the 910A fleet.

DLO's default weight transfer is AllGather. The hand-rolled ``HcclAllGatherV``
path failed with HCCL_E_NOT_SUPPORT on this hardware, which pushed the
vLLM-Omni offload path onto rank-local copies (8x host RAM). If fixed-size
``all_gather`` works, the shared-weight path is viable again.

    torchrun --nproc_per_node=8 probe_hccl_collectives.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def try_op(name, fn):
    try:
        out = fn()
        torch.npu.synchronize()
        print(f"OK   {name}", flush=True)
        return out
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0][:130]
        print(f"FAIL {name}: {type(exc).__name__}: {msg}", flush=True)
        return None


def main() -> int:
    # torchrun does not pin the NPU per rank the way it does for CUDA, so bind
    # explicitly before creating the HCCL communicator.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible:
        local_rank = int(visible.split(",")[local_rank])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.npu.current_device()
    if rank == 0:
        print(f"world={world} backend=hccl", flush=True)

    def all_gather_fixed():
        tensor = torch.full((1024,), float(rank), dtype=torch.float16, device=f"npu:{device}")
        bucket = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(bucket, tensor)
        return bucket

    def all_gather_into_tensor():
        tensor = torch.full((1024,), float(rank), dtype=torch.float16, device=f"npu:{device}")
        out = torch.empty(world * 1024, dtype=torch.float16, device=f"npu:{device}")
        dist.all_gather_into_tensor(out, tensor)
        return out

    def all_gather_variable():
        sizes = [1024 + i for i in range(world)]
        tensor = torch.full((sizes[rank],), float(rank), dtype=torch.float16, device=f"npu:{device}")
        bucket = [torch.empty(sizes[i], dtype=torch.float16, device=f"npu:{device}") for i in range(world)]
        dist.all_gather(bucket, tensor)
        return bucket

    def broadcast_op():
        tensor = torch.full((1024,), float(rank), dtype=torch.float16, device=f"npu:{device}")
        dist.broadcast(tensor, src=0)
        return tensor

    def all_reduce_op():
        tensor = torch.ones(1024, dtype=torch.float16, device=f"npu:{device}")
        dist.all_reduce(tensor)
        return tensor

    results = {}
    for name, fn in (
        ("all_gather [list, fixed]", all_gather_fixed),
        ("all_gather_into_tensor", all_gather_into_tensor),
        ("all_gather [list, variable]", all_gather_variable),
        ("broadcast", broadcast_op),
        ("all_reduce", all_reduce_op),
    ):
        results[name] = try_op(name, fn) is not None

    if rank == 0:
        print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in results.items()), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())