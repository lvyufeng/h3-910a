#!/usr/bin/env python3
"""R244 driver: one measured step; account the block-boundary weight wait."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_REST244_LABEL", "rest244a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401
    import budget_probe244 as probe

    torch.npu.set_device(local_rank)
    patch_report = probe.install(torch)

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
    data["rest244"] = probe.collect(torch)
    data["rest244_patch"] = patch_report
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["rest244_invariance"] = {
        "measured_rank_max_s": measured, "reference_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
        "accepted": None if measured is None else abs(measured - 17.928) <= 0.10,
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured,
                      "wait": data["rest244"].get("measured_like"),
                      "all": data["rest244"].get("all"),
                      "samples": data["rest244"]["event_samples"],
                      "type": data["rest244"]["ready_event_type"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
