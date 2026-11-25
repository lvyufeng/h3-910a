#!/usr/bin/env python3
"""R247 driver: R218 FA padding-task trim, optionally combined with the 64/912 tiles.

User approved (2026-11-25) both the R218 trim and the FA tile change. This run measures
the trim alone (128/304) and the combination (64/912), each as its own single-step case.
Nothing on disk is modified: the deployed backend's module-level ``flash_attention_910a``
is replaced in-process by the R218 trim implementation, which the consumer resolves late
(``from . import fa910a as _fa; _fa.flash_attention_910a(...)``), so the swap is safe and
scoped to this process.

Env (set by the launcher): H3_FA_TRIM_Q_LIMIT, H3_FA_BR_CAP, H3_FA_BC_CAP,
H3_FA215_KERNEL_DIR.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_TRIM247_LABEL", "trim247a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)

    from vllm_omni.diffusion.attention.backends import fa910a as deployed
    import bb218trim as trim_mod

    br_cap = os.environ.get("H3_FA_BR_CAP", "128")
    bc_cap = os.environ.get("H3_FA_BC_CAP", "304")
    q_limit = os.environ.get("H3_FA_TRIM_Q_LIMIT", "")
    deployed.flash_attention_910a = trim_mod.flash_attention_910a

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
    data["trim247"] = {
        "case": LABEL,
        "fa_impl": "R218 padding-task trim",
        "br_cap": int(br_cap), "bc_cap": int(bc_cap),
        "trim_q_limit": int(q_limit) if q_limit else None,
        "kernel_dir": os.environ.get("H3_FA215_KERNEL_DIR"),
        "deployed_source_changed": False,
        "user_approved": ["R218 FA trim", "FA tile 64/912"],
    }
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["trim247_invariance"] = {
        "measured_rank_max_s": measured, "baseline_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    ch = (data.get("checks") or [{}])[0]
    print(json.dumps({"rank": rank, "step_s": measured,
                      "br": int(br_cap), "bc": int(bc_cap),
                      "video_relRMS": (ch.get("video_vs_R124_baseline") or {}).get(
                          "relative_RMS_vs_R124_NPU_baseline")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
