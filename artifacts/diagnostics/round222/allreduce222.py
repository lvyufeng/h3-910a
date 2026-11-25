#!/usr/bin/env python3
"""R222b: measure the row-parallel TP all-reduce on a real TP2 HCCL group.

R221 left ~11.2/11.5 ms per guarded call unaccounted between standalone and
in-model measurements, and R222 disproved the "host sync drains the prefetch
stream" hypothesis. The remaining structural candidate is the row-parallel
all-reduce of the FP32 activation (10448 x 5376 x 4 B = 225 MB per call, two
per block, 50 blocks).

No weights, no model, no source change. Launch with:
  torch_npu_run --standalone --nproc-per-node 2 allreduce222.py --out ...
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--m", type=int, default=10448)
    ap.add_argument("--n", type=int, default=5376)
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401
    import torch.distributed as dist

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    props = torch.npu.get_device_properties(rank)
    torch.manual_seed(222)  # identical on every rank/config
    x_init = torch.randn(args.m, args.n, dtype=torch.float32, device=device)
    x = x_init.clone()
    torch.npu.synchronize()
    dist.barrier()

    def once(kind):
        if kind == "all_reduce":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif kind == "all_gather":
            out = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(out, x)
        else:
            raise ValueError(kind)

    samples = {}
    for kind in ("all_reduce", "all_gather"):
        for _ in range(args.warmup):
            once(kind)
        torch.npu.synchronize()
        values = []
        for _ in range(args.steps):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            once(kind)
            end.record()
            torch.npu.synchronize()
            values.append(start.elapsed_time(end))
        samples[kind] = values

    nbytes = args.m * args.n * 4
    ar_ms = statistics.median(samples["all_reduce"])
    # ring all-reduce moves 2*(world-1)/world * bytes through the link
    moved = 2 * (world - 1) / world * nbytes
    report = {
        "schema_version": "r222b.tp_allreduce.v1",
        "host": "ts-217",
        "measurement_only": True,
        "no_model_weights_loaded": True,
        "no_source_modification": True,
        "rank": rank, "world_size": world,
        "device": {"logical": rank,
                   "name": str(getattr(props, "name", "unknown"))},
        "shape": {"m": args.m, "n": args.n, "dtype": "float32", "bytes": nbytes},
        "all_reduce_ms": {"n": len(samples["all_reduce"]),
                          "min": min(samples["all_reduce"]),
                          "median": ar_ms, "max": max(samples["all_reduce"])},
        "all_gather_ms": {"n": len(samples["all_gather"]),
                          "min": min(samples["all_gather"]),
                          "median": statistics.median(samples["all_gather"]),
                          "max": max(samples["all_gather"])},
        "all_reduce_link_GB_per_s": round(moved / (ar_ms * 1e-3) / 1e9, 1),
        "per_step_projection": {
            "calls_per_step": 100,
            "ms_per_step_if_serial": round(ar_ms * 100, 1),
            "note": "two row-parallel reductions per block x 50 blocks; a projection, "
                    "not a measurement of the composed model",
        },
    }
    import hashlib
    import os
    x_check = x_init.clone()
    dist.all_reduce(x_check, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    digest = None
    if rank == 0:
        digest = hashlib.sha256(x_check.cpu().numpy().tobytes()).hexdigest()
    del x_check
    report["reduced_result_sha256_rank0"] = digest
    report["hccl_env"] = {key: os.environ[key] for key in
                          ("HCCL_ALG", "HCCL_BUFFSIZE", "HCCL_DETERMINISTIC",
                           "HCCL_IF_BASE_PORT", "HCCL_CONNECT_TIMEOUT")
                          if key in os.environ}
    if rank == 0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print("RESULT " + json.dumps({"ar_ms": round(ar_ms, 3),
                                      "link_GBps": report["all_reduce_link_GB_per_s"],
                                      "projected_s_per_step": round(ar_ms * 100 / 1000, 3),
                                      "sha256_rank0": digest,
                                      "env": report["hccl_env"]}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())