"""Validate a four-NPU HCCL process group with real AllReduce traffic."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def hccl_build_version() -> str:
    try:
        import torch_npu

        hccl = getattr(getattr(torch_npu, "version", None), "hccl", None)
        if hccl is not None:
            return str(hccl)
    except Exception:
        pass
    return "unknown"


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")

    if world_size != args.world_size:
        raise RuntimeError(f"world_size={world_size}, expected={args.world_size}")
    if torch.npu.device_count() != args.world_size:
        raise RuntimeError(
            f"visible NPU count={torch.npu.device_count()}, expected={args.world_size}; "
            f"ASCEND_RT_VISIBLE_DEVICES={visible}"
        )

    torch.npu.set_device(local_rank)
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    try:
        value = torch.tensor(
            [[float(rank + 1), float((rank + 1) * 10)]],
            dtype=torch.float16,
            device=f"npu:{local_rank}",
        )
        expected_values = [
            float(sum(range(1, world_size + 1))),
            float(sum(i * 10 for i in range(1, world_size + 1))),
        ]
        expected = torch.tensor(
            [expected_values],
            dtype=torch.float16,
            device=f"npu:{local_rank}",
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()

        gathered = [torch.empty_like(value) for _ in range(world_size)]
        dist.all_gather(gathered, value)
        torch.npu.synchronize()

        if not torch.equal(value, expected):
            raise RuntimeError(
                f"rank {rank}: all_reduce={value.cpu().tolist()}, "
                f"expected={expected.cpu().tolist()}"
            )
        for peer, tensor in enumerate(gathered):
            if not torch.equal(tensor, expected):
                raise RuntimeError(
                    f"rank {rank}: all_gather peer {peer}={tensor.cpu().tolist()}, "
                    f"expected={expected.cpu().tolist()}"
                )

        result = {
            "instance": args.instance,
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "visible_devices": visible,
            "hccl_build_version": hccl_build_version(),
            "all_reduce": value.cpu().tolist(),
            "all_gather_ok": True,
            "ok": True,
        }
        print("HCCL_PROBE " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
