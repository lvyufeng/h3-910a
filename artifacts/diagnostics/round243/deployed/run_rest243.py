#!/usr/bin/env python3
"""R243 driver: one measured step; block-boundary weight-wait accounting."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_REST243_LABEL", "rest243a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401
    import budget_probe243 as probe

    torch.npu.set_device(local_rank)
    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    import h3_replay123.transformer as T  # noqa: F401

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
        "--mode", "baseline", "--label", LABEL,
        "--tp", "2", "--usp", "4", "--offload", "native",
        "--audit-sequence", "strict",
        "--steps", "0", "--warmup", "1", "--repeats", "1",
    ]
    import benchmark_actual_attention126 as bench

    bench.ROOT = ROOT
    bench.main()

    receipt = out / f"candidate_rank{rank}.json"
    data = json.loads(receipt.read_text()) if receipt.exists() else {}
    data["rest243"] = probe.collect()
    data["rest243_hooks"] = hook_report
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["rest243_invariance"] = {
        "measured_rank_max_s": measured, "reference_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
        "accepted": None if measured is None else abs(measured - 17.928) <= 0.15,
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    summary = data["rest243"]
    print(json.dumps({"rank": rank, "step_s": measured,
                      "hooks": hook_report,
                      "measured_step": summary.get("measured_like"),
                      "warmup_like": summary.get("warmup_like")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
