#!/usr/bin/env python3
"""R227 driver: one measured step with copy-stream timing of the layerwise prefetch.

Measurement only: no math, dtype, block-count, offload-policy or scheduling change;
no candidate installed. Answers whether the 3.37 s/step attn+mlp remainder measured
in R226 is exposed weight-prefetch wait.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_REST227_LABEL", "rest227a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401
    import budget_probe227 as probe

    torch.npu.set_device(local_rank)

    import vllm_omni.diffusion.offloader.layerwise_backend as lb

    hook_patch = probe.install_offloader_probe(torch, lb.LayerwiseOffloadHook)

    real_group = lb._install_layerwise_hook_group
    span_report = {}

    def patched(blocks, device, stream, pin_memory=True):
        hooks = real_group(blocks, device, stream, pin_memory)
        span_report.update(probe.install(torch, blocks))
        return hooks

    lb._install_layerwise_hook_group = patched

    sys.argv = [
        "benchmark_actual_attention126.py",
        "--checkpoint", "/home/mseco/models/MiniMax-H3/FL2VA/transformer",
        "--mode", "baseline",
        "--label", LABEL,
        "--tp", "2", "--usp", "4", "--offload", "native",
        "--audit-sequence", "strict",
        "--steps", "0", "--warmup", "1", "--repeats", "1",
    ]
    import benchmark_actual_attention126 as bench

    bench.ROOT = ROOT
    bench.main()

    receipt = out / f"candidate_rank{rank}.json"
    data = json.loads(receipt.read_text()) if receipt.exists() else {}
    data["rest227"] = probe.collect(torch)
    data["rest227_hook_patch"] = hook_patch
    data["rest227_spans_installed"] = span_report
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank,
                      "copy_ms_sum": data["rest227"]["copy_stream"]["copy_ms_sum_per_step"],
                      "sync_fallback": data["rest227"]["synchronous_fallback"]["count"],
                      "spans": {k: v["median_ms"] for k, v in data["rest227"]["spans"].items()}}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())