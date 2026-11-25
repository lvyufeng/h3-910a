# R245: one (br_cap, bc_cap) point of the FA tile sweep, with paired numeric comparison.
#!/usr/bin/env python3
"""R245: one FA tile point, seeded inputs, timing + numeric drift against a baseline output.

Approved scope (user 2026-11-25): standalone FA microbenchmark only. Every tile change
alters the accumulation order, so these points are *measurements*, not candidates. No model
is loaded, no 49-step artifact is produced, and the default stays 128/304.

Inputs are seeded identically for every point (torch.manual_seed(20261124)), the real shape is
B=1,H=7,N=41792,D=128. When H3_FA_REF_NPY points at a baseline output, this point also reports
max|diff| and relRMS against it so the numeric cost of each tile is explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

BR_CAP = int(os.environ['H3_FA_BR_CAP'])
BC_CAP = int(os.environ['H3_FA_BC_CAP'])
OUT = Path(os.environ['H3_FA_SWEEP_OUT'])
B, H, N, D = 1, 7, 41792, 128
WARMUP, REPS = 2, 5


def main() -> int:
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    torch.manual_seed(20261124)
    sys.path.insert(0, os.environ.get("H3_FA_PROBE_DIR", "/tmp/h3_fa216"))
    import fa910a_sweep as fa

    torch.npu.set_device(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')

    t0 = time.perf_counter()
    out = fa.flash_attention_910a(q, k, v)
    torch.npu.synchronize()
    build_s = time.perf_counter() - t0

    for _ in range(WARMUP):
        fa.flash_attention_910a(q, k, v)
    torch.npu.synchronize()

    times = []
    for _ in range(REPS):
        torch.npu.synchronize()
        a = time.perf_counter()
        fa.flash_attention_910a(q, k, v)
        torch.npu.synchronize()
        times.append((time.perf_counter() - a) * 1000.0)
    med = statistics.median(times)
    best = min(times)

    kernel_n = fa._kernel_sequence_length(N, D, enabled=fa._USE_TILE_PADDING)
    br = fa.pick_tile(kernel_n, BR_CAP)
    bc = fa.pick_tile(kernel_n, BC_CAP)
    qb = (kernel_n + br - 1) // br
    tiles = sum(math.ceil(((i + 1) * br) / bc) for i in range(qb))
    flops = tiles * 4 * br * bc * D * H

    out_f32 = out.float()
    row = {
        'br_cap': BR_CAP, 'bc_cap': BC_CAP, 'br': br, 'bc': bc,
        'tiles': tiles, 'causal_TFLOP': round(flops / 1e12, 3),
        'build_s': round(build_s, 2),
        'median_ms': round(med, 2), 'best_ms': round(best, 2),
        'achieved_TFLOPS_per_card': round(flops / (med / 1000.0) / 1e12, 2),
        'output_finite': bool(torch.isfinite(out).all().item()),
        'output_sum': float(out_f32.sum().item()),
        'output_sha256': hashlib.sha256(
            out.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'times_ms': [round(x, 3) for x in times],
    }

    save = os.environ.get("H3_FA_SAVE_NPY")
    if save:
        np.save(save, out.detach().cpu().contiguous().numpy())
        row['saved_npy'] = save

    ref = os.environ.get("H3_FA_REF_NPY")
    if ref and os.path.exists(ref):
        reference = np.load(ref)
        got = out.detach().cpu().contiguous().numpy()
        diff = np.abs(got.astype(np.float32) - reference.astype(np.float32))
        denom = np.sqrt(np.mean(reference.astype(np.float32) ** 2))
        row['vs_baseline'] = {
            'ref': ref,
            'max_abs_diff': float(diff.max()),
            'rel_rms': float(np.sqrt(np.mean(diff ** 2)) / denom),
            'bit_identical': bool((got == reference).all()),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({k: row[k] for k in ('br', 'bc', 'median_ms', 'achieved_TFLOPS_per_card')} |
                     ({'vs_baseline': row['vs_baseline']} if 'vs_baseline' in row else {})), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
