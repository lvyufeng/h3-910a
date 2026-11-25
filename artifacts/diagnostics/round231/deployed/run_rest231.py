#!/usr/bin/env python3
"""R231 driver: one measured step with function-level probes inside attention/MLP.

Adds device events only: module hooks plus patches of ``_apply_rope``,
``MiniMaxH3Attention._run_packed_attention`` and ``F.silu``. No math, dtype, block
count, offload policy or scheduling change; no candidate installed.

The step time must stay within 0.1 s of 17.93 s for the run to be accepted; the
driver records the measured step so the caller can enforce that.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/tmp/h3_budget149")
LABEL = os.environ.get("H3_REST231_LABEL", "rest231a")


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    out = ROOT / LABEL
    assert out.is_dir(), "launcher must pre-create the output directory"
    sys.path.insert(0, str(ROOT))

    import torch
    import torch_npu  # noqa: F401
    import budget_probe231 as probe

    torch.npu.set_device(local_rank)

    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    import h3_replay123.transformer as transformer_module

    fn_report = probe.install_function_probes(torch, transformer_module)

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
    data["rest231"] = probe.collect(torch)
    data["rest231_function_report"] = fn_report
    data["rest231_spans_installed"] = span_report
    measured = (data.get("checks") or [{}])[0].get("rank_max_elapsed_s")
    data["rest231_step_invariance"] = {
        "measured_rank_max_s": measured,
        "reference_s": 17.928,
        "delta_s": None if measured is None else round(measured - 17.928, 3),
        "accepted": None if measured is None else abs(measured - 17.928) <= 0.1,
        "note": "R227 showed a 15% perturbation is possible; only accepted runs may be quoted",
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n")
    labels = {k: v["median_ms"] for k, v in data["rest231"]["labels"].items()}
    print(json.dumps({"rank": rank, "step_s": measured,
                      "invariance_delta_s": data["rest231_step_invariance"]["delta_s"],
                      "labels": labels}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())