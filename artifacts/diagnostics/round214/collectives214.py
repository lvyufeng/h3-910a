#!/usr/bin/env python3
"""R214: complete the collective determinism sweep - all_reduce AND all_to_all AND reduce_scatter AND all_gather.

R210 only tested all_reduce, which is a real gap: USP with --usp 4 relies heavily on all-to-all.
Same methodology: identical inputs, repeated, bit-compared.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

OUT_DIR = Path('/tmp/h3_coll214')
REPEATS = 10


def word_hash(t) -> str:
    return hashlib.sha256(t.detach().cpu().float().numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ.get('LOCAL_RANK', rank))
    torch.npu.set_device(local)
    dist.init_process_group(backend='hccl')

    rows = []

    def record(op, hashes, extra=None):
        uniq = len(set(hashes))
        row = {'op': op, 'repeats': len(hashes), 'distinct_results': uniq,
               'reproducible': uniq == 1, 'first_three': hashes[:3]}
        if extra:
            row.update(extra)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

    # --- 1. all_reduce (already known reproducible from R210, re-checked as a control)
    n = 5120
    torch.manual_seed(21)
    base = torch.randn(n, dtype=torch.float32, device=f'npu:{local}')
    hs = []
    for _ in range(REPEATS):
        buf = base.clone()
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        dist.barrier()
        hs.append(word_hash(buf))
    record('all_reduce_sum', hs)

    # --- 2. all_to_all_single: the USP primitive I missed in R210
    # 8 ranks, split 5120 into 8 chunks of 640 and shuffle
    torch.manual_seed(22)
    chunk = n // world
    out_hashes = []
    for _ in range(REPEATS):
        inp = base.clone()                       # identical input every repeat
        out = torch.empty_like(inp)
        dist.all_to_all_single(out, inp, output_split_sizes=[chunk] * world,
                              input_split_sizes=[chunk] * world)
        dist.barrier()
        out_hashes.append(word_hash(out))
    record('all_to_all_single', out_hashes)

    # --- 3. reduce_scatter: another primitive the model may use
    torch.manual_seed(23)
    rs_in = torch.randn(n, dtype=torch.float32, device=f'npu:{local}')
    rs_hashes = []
    for _ in range(REPEATS):
        out = torch.empty(chunk, dtype=torch.float32, device=f'npu:{local}')
        dist.reduce_scatter_tensor(out, rs_in.clone(), op=dist.ReduceOp.SUM)
        dist.barrier()
        rs_hashes.append(word_hash(out))
    record('reduce_scatter', rs_hashes)

    # --- 4. all_gather: used by TP row-parallel / USP gather
    torch.manual_seed(24)
    ag_base = torch.randn(chunk, dtype=torch.float32, device=f'npu:{local}')
    ag_hashes = []
    for _ in range(REPEATS):
        # FIX (R214 self-error): the first version called torch.randn inside the loop, so every
        # repeat got different input data and the "non-reproducible" verdict was my bug, not HCCL's.
        inp = ag_base.clone()
        out = torch.empty(n, dtype=torch.float32, device=f'npu:{local}')
        dist.all_gather_into_tensor(out, inp)
        dist.barrier()
        ag_hashes.append(word_hash(out))
    record('all_gather', ag_hashes)

    report = {'world_size': world, 'rows': rows,
              'all_reproducible': all(r['reproducible'] for r in rows)}
    gathered = [None] * world
    dist.all_gather_object(gathered, report)
    dist.barrier()
    if rank == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / 'collectives214.json').write_text(json.dumps(gathered, indent=2) + '\n')
        print('VERDICT', 'all_reproducible' if report['all_reproducible'] else 'NON-DET FOUND', flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
