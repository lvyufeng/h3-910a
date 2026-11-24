# R215: the previously unmeasured Br=64/Bc=768 combination, adjacent to the R194 boundary.
#!/usr/bin/env python3
"""R191: one (br_cap, bc_cap) point of the FA tile sweep.

Motivation (R190): the FA tile costs 6.00 us against a 1.44 us cube floor, and cube, vector and
HBM all sit near 20% utilisation. One candidate explanation is cube fragmentation because Bc=304
is 2x128+48. This measures whether a Bc that is a clean multiple of 128 is faster.

Measurement only: changing the tile changes accumulation order, so nothing measured here is a
candidate and the default stays 128/304.
"""
from __future__ import annotations
import json
import os
import statistics
import sys
import time
from pathlib import Path

BR_CAP = int(os.environ['H3_FA_BR_CAP'])
BC_CAP = int(os.environ['H3_FA_BC_CAP'])
OUT = Path(os.environ['H3_FA_SWEEP_OUT'])
B, H, N, D = 1, 7, 41792, 128      # the shape the model actually uses
WARMUP, REPS = 2, 5


def main():
    import torch
    import torch_npu  # noqa: F401
    import hashlib
    torch.manual_seed(20261124)  # identical inputs for every isolated comparison
    sys.path.insert(0, os.environ.get("H3_FA_PROBE_DIR", "/tmp/h3_fa216"))
    import fa910a_sweep as fa

    torch.npu.set_device(0)
    q = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    k = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')
    v = torch.randn((B, H, N, D), dtype=torch.float16, device='npu')

    # first call builds the TIK kernel; time that separately so the build is not in the timing
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

    # Compile-time geometry is selected after optional sequence padding. Reporting it
    # from the input N alone mis-states every padded case (R215 caught 64/64 in the
    # old probe while the launch log correctly showed 64/768).
    import math
    kernel_n = fa._kernel_sequence_length(N, D, enabled=fa._USE_TILE_PADDING)
    br = fa.pick_tile(kernel_n, BR_CAP)
    bc = fa.pick_tile(kernel_n, BC_CAP)
    qb, kb_t = (kernel_n + br - 1) // br, (kernel_n + bc - 1) // bc
    tiles = sum(math.ceil(((i + 1) * br) / bc) for i in range(qb))
    flops = tiles * 4 * br * bc * D * H

    row = {
        'br_cap': BR_CAP, 'bc_cap': BC_CAP, 'br': br, 'bc': bc,
        'tiles': tiles, 'causal_TFLOP': round(flops / 1e12, 3),
        'build_s': round(build_s, 2),
        'median_ms': round(med, 2), 'best_ms': round(best, 2),
        'achieved_TFLOPS_per_card': round(flops / (med / 1000.0) / 1e12, 2),
        'output_finite': bool(torch.isfinite(out).all().item()),
        'output_sum': float(out.float().sum().item()),
        'output_sha256': hashlib.sha256(
            out.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        'times_ms': [round(x, 3) for x in times],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=1))


if __name__ == '__main__':
    main()
