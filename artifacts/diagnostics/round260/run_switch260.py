#!/usr/bin/env python3
"""R260 driver: validate the landed switch module itself (H3_PERF_COMBO=1)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_SWITCH260_LABEL", "switch260a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)

    import h3_perf_switches as perf

    perf_report = perf.install(torch, label=LABEL)

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
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["switch260"] = {
        "path": "artifacts/deploy/h3_perf_switches.py (H3_PERF_COMBO=1)",
        "install_report": perf_report,
        "deployed_source_changed": False,
        "expected_s": 16.32,
    }
    data["switch260_delta_vs_expected"] = (
        None if measured is None else round(measured - 16.32, 3))
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured,
                      "applied": perf_report.get("applied"),
                      "skipped": perf_report.get("skipped")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
