#!/usr/bin/env python3
"""R226 driver: one measured step with a device-event split of the ``rest`` region.

R223 sized the unmeasured remainder at 4.121 s/step (45% of a 17.9 s step is
non-compute). This run installs event hooks only; it changes no math, dtype,
block count, offload policy or scheduling, and installs no candidate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_REST226_LABEL", "rest226b")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401
    import budget_probe226 as probe

    torch.npu.set_device(local_rank)
    import vllm_omni.diffusion.offloader.layerwise_backend as lb

    real_group = lb._install_layerwise_hook_group
    hook_report = {}

    def patched(blocks, device, stream, pin_memory=True):
        hooks = real_group(blocks, device, stream, pin_memory)
        hook_report.update(probe.install(torch, blocks))
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
    data["rest226"] = probe.collect(torch)
    data["rest226_hook_install"] = hook_report
    data["rest226_scope"] = (
        "single measured step, device events only, no math/dtype/block/scheduling change; "
        "label sums are per-step over 50 blocks and must not be added to parent spans")
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "rest226": data["rest226"].get("labels", {})}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())