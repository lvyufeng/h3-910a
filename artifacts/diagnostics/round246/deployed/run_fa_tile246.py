#!/usr/bin/env python3
"""R246 driver: in-model measurement of the user-approved FA tile 64/912.

User approved (2026-11-25) adopting FA tiles 64/912 (standalone 1.12x, relRMS~2e-4).
This driver does NOT modify deployed sources: it monkeypatches the FA backend's
``_build`` so every kernel instance uses br_cap=64, bc_cap=912. One measured step,
strict audit, baseline numerics otherwise untouched.

Everything else (blocks, steps, dtypes, FP32 exceptions, 155 Half checks, TP2+USP4,
offload schedule) is unchanged.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_TILE246_LABEL", "tile246a")
BR_CAP = int(os.environ.get("H3_TILE246_BR", "64"))
BC_CAP = int(os.environ.get("H3_TILE246_BC", "912"))


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(local_rank)

    from vllm_omni.diffusion.attention.backends import fa910a

    real_build = fa910a._build
    seen = []

    def build_with_tiles(b, h, n, d, br_cap=BR_CAP, bc_cap=BC_CAP, key_seq_limit=None):
        br = fa910a.pick_tile(n, BR_CAP)
        bc = fa910a.pick_tile(n, BC_CAP)
        seen.append({"b": b, "h": h, "n": n, "d": d, "br": br, "bc": bc})
        return real_build(b, h, n, d, BR_CAP, BC_CAP, key_seq_limit)

    fa910a._build = build_with_tiles

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
    data["tile246"] = {
        "br_cap": BR_CAP, "bc_cap": BC_CAP,
        "kernels_built_instances": seen[:4],
        "instances_seen": len(seen),
        "patched": "fa910a._build injects user-approved 64/912 tiles",
        "deployed_source_changed": False,
    }
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["tile246_invariance"] = {
        "measured_rank_max_s": measured, "baseline_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
        "expected_saving_s": 0.95,
        "expected_s": 16.98,
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"rank": rank, "step_s": measured,
                      "tiles": data["tile246"]["kernels_built_instances"],
                      "instances": len(seen)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
