#!/usr/bin/env python3
"""R210: is HCCL all-reduce bit-reproducible for identical inputs?

R209 hypothesised that non-deterministic collective reduction order explains why the encoder (96% of
words differing), the DiT (relRMS 0.026%) and the VAE (1 LSB) are all non-repeatable. This tests the
hypothesis directly with no model: identical input, repeated all-reduce, bit-compared, and then the
same probe again with HCCL_DETERMINISTIC=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

OUT_DIR = Path('/tmp/h3_hccl210')
REPEATS = 12


def word_hash(t) -> str:
    raw = t.detach().cpu().float().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ.get('LOCAL_RANK', rank))
    torch.npu.set_device(local)
    dist.init_process_group(backend='hccl')

    report = {'world_size': world, 'repeats': REPEATS,
              'HCCL_DETERMINISTIC': os.environ.get('HCCL_DETERMINISTIC', '<unset>'),
              'rank': rank}

    rows = []
    # several shapes, including the real per-layer widths the model uses
    for tag, n in (('small', 4096), ('hidden5120', 5120), ('inter25600', 25600)):
        torch.manual_seed(1234 + n)          # identical input on every rank and every repeat
        base = torch.randn(n, dtype=torch.float32, device=f'npu:{local}')
        hashes = []
        for _ in range(REPEATS):
            buf = base.clone()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            dist.barrier()
            hashes.append(word_hash(buf))
        uniq = len(set(hashes))
        rows.append({'shape': tag, 'n': n, 'distinct_results': uniq,
                     'reproducible': uniq == 1, 'first_three': hashes[:3]})
        print(json.dumps(rows[-1]), flush=True)

    report['rows'] = rows
    report['all_reproducible'] = all(r['reproducible'] for r in rows)
    report['verdict'] = ('all-reduce IS bit-reproducible for identical inputs'
                         if report['all_reproducible']
                         else 'all-reduce is NOT bit-reproducible for identical inputs')

    gathered = [None] * world
    dist.all_gather_object(gathered, report)
    dist.barrier()
    if rank == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"hccl_{os.environ.get('HCCL_DETERMINISTIC', 'unset')}.json"
        path.write_text(json.dumps({'tag': os.environ.get('HCCL_DETERMINISTIC', 'unset'),
                                    'ranks': gathered}, indent=2) + '\n')
        print('WROTE', path, flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
