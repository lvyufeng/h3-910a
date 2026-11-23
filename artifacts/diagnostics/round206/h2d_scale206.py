#!/usr/bin/env python3
"""R206: does H2D bandwidth hold at the encoder's real 63 GB scale?

R198 measured pinned H2D at only up to 2 GB and got 20.0 GB/s, then R203/R205 used that figure to
derive the encoder's 3.15 s floor. If bandwidth degrades at 63 GB scale, that floor is wrong.
This measures pinned bandwidth at 8/16/32 GB, in the chunked pattern a stager would use.
"""
from __future__ import annotations
import json
import statistics
import time
from pathlib import Path

OUT = Path('/tmp/h3_h2d206/h2d_scale206.json')
SIZES_GB = [8, 16, 32]
CHUNK_GB = 1.0          # stream in 1 GB chunks, like a stager would


def main():
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)

    rows = []
    chunk_n = int(CHUNK_GB * 1024**3 / 2)      # fp16 words
    for gb in SIZES_GB:
        try:
            total_n = int(gb * 1024**3 / 2)     # fp16 words
            host = torch.empty(total_n, dtype=torch.float16, pin_memory=True)
            dev = torch.empty(total_n, dtype=torch.float16, device='npu')
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for off in range(0, total_n, chunk_n):
                end = min(off + chunk_n, total_n)
                dev[off:end].copy_(host[off:end], non_blocking=True)
            torch.npu.synchronize()
            dt = time.perf_counter() - t0
            rows.append({'gb': gb, 'chunk_GB': CHUNK_GB, 'seconds': round(dt, 3),
                         'GB_per_s': round(gb / dt, 2)})
            print(json.dumps(rows[-1]), flush=True)
            del host, dev
            torch.npu.empty_cache()
        except Exception as exc:  # noqa: BLE001
            rows.append({'gb': gb, 'error': repr(exc)[:200]})
            print(json.dumps(rows[-1]), flush=True)

    # extrapolate the encoder's 63 GB transfer at the measured rates
    ok = [r for r in rows if 'GB_per_s' in r]
    out = {'scope': 'R206: pinned H2D bandwidth at the encoder real 63 GB scale',
           'rows': rows, 'encoder_weights_GB': 63.0,
           'extrapolation': [{'gb_scale': r['gb'], 'GB_per_s': r['GB_per_s'],
                              'implied_63GB_seconds': round(63.0 / r['GB_per_s'], 2)} for r in ok]}
    if ok:
        worst = min(r['GB_per_s'] for r in ok)
        out['worst_case_GB_per_s'] = worst
        out['worst_case_63GB_seconds'] = round(63.0 / worst, 2)
        print(json.dumps({'worst': worst, 'sixty_three_GB_seconds': out['worst_case_63GB_seconds']}, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
