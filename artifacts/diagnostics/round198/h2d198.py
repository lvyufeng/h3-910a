#!/usr/bin/env python3
"""R198b: measure host-to-device bandwidth, to size the VAE residency shuttling.

R198's primary suspicion is that the video VAE re-shuttles its ~10.4 GB of FP32 parameters once per
tile (4 tiles per rank are run serially). A single H2D bandwidth number turns that into seconds.
"""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

OUT = Path('/tmp/h3_h2d198/h2d198.json')
SIZES_GB = [0.25, 1.0, 2.0]


def bench(gb, pinned):
    import torch
    n = int(gb * 1024**3 / 4)          # fp32 words
    cpu = torch.empty(n, dtype=torch.float32, pin_memory=pinned)
    npu = torch.empty(n, dtype=torch.float32, device='npu')
    ts = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        npu.copy_(cpu, non_blocking=True)
        torch.npu.synchronize()
        ts.append(time.perf_counter() - t0)
    del cpu, npu
    torch.npu.empty_cache()
    med = statistics.median(ts)
    return {'gb': gb, 'pinned': pinned, 'median_s': round(med, 4),
            'GB_per_s': round(gb / med, 1)}


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    rows = []
    for gb in SIZES_GB:
        for pinned in (False, True):
            try:
                rows.append(bench(gb, pinned))
            except Exception as exc:  # noqa: BLE001
                rows.append({'gb': gb, 'pinned': pinned, 'error': repr(exc)[:160]})
            print(json.dumps(rows[-1]), flush=True)

    # implications for the VAE: 10.4 GB of FP32 parameters, 4 tiles per rank, 8 ranks
    best = max((r for r in rows if 'GB_per_s' in r), key=lambda r: r['GB_per_s'], default=None)
    out = {'scope': 'R198b: host-to-device bandwidth to size VAE residency shuttling',
           'rows': rows, 'best_GB_per_s': best['GB_per_s'] if best else None,
           'vae_parameters_GB': 10.4, 'tiles_per_rank': 4, 'ranks': 8}
    if best:
        bw = best['GB_per_s']
        out['implications'] = {
            'one_full_shuttle_s_per_rank': round(10.4 / bw, 2),
            'four_tiles_full_shuttle_s_per_rank': round(4 * 10.4 / bw, 2),
            'if_all_8_ranks_shuttle_concurrently_note':
                'H2D bandwidth is typically shared, so concurrent shuttling can be slower than one rank alone',
        }
        print(json.dumps(out['implications'], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
