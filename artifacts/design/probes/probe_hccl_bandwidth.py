#!/usr/bin/env python3
"""Measure HCCL collective bandwidth on this 910A box.

The DiT profile shows ~14s per block at ~1% of usable FLOPs, with the first ten
blocks nearly free and the cost growing afterwards. That points at weight
movement (HSDP all-gather per block) rather than arithmetic. Before rewriting
anything, measure the collective itself: if all-gather only reaches tens of MB/s
the problem is the HCCL configuration, and fixing it is far cheaper than
re-architecting the model's parallelism.

Run with torchrun over eight ranks.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist


def _bench(fn, warmup: int = 2, iters: int = 5) -> float:
    """Return median seconds per call."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")

    # One DiT block is ~1.24 GiB in bf16; HSDP all-gathers that once per block.
    for mb in (16, 256, 1240):
        elems = mb * 1024 * 1024 // 2
        # bf16 collectives are unreliable on 910A (HcclAllreduce parameter
        # error), so benchmark in fp16 which is what the serving path uses.
        tensor = torch.ones(elems, dtype=torch.float16, device=f"npu:{local_rank}").contiguous()

        def all_gather(t=tensor):
            out = torch.empty(world * t.numel(), dtype=t.dtype, device=t.device)
            dist.all_gather_into_tensor(out, t)

        def all_reduce(t=tensor):
            dist.all_reduce(t.contiguous())

        def broadcast(t=tensor):
            dist.broadcast(t, src=0)

        ag = _bench(all_gather)
        ar = _bench(all_reduce)
        bc = _bench(broadcast)

        if rank == 0:
            # all_gather moves (world-1) * size of payload per rank.
            ag_bw = (mb * (world - 1) * 1024 * 1024) / ag / 1e6
            ar_bw = (2 * (world - 1) / world * mb * 1024 * 1024) / ar / 1e6
            bc_bw = (mb * 1024 * 1024) / bc / 1e6
            print(
                f"{mb:>5} MiB  "
                f"all_gather {ag * 1e3:8.2f} ms ({ag_bw:7.1f} MB/s)  "
                f"all_reduce {ar * 1e3:8.2f} ms ({ar_bw:7.1f} MB/s)  "
                f"broadcast {bc * 1e3:8.2f} ms ({bc_bw:7.1f} MB/s)",
                flush=True,
            )

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())