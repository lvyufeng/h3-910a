#!/usr/bin/env python3
"""R185 runner: one 8-rank replay, identical except for H3_FAST_AUDIT."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path('/tmp/h3_fast185')
LABEL = os.environ['H3_FAST185_LABEL']
STEPS = os.environ.get('H3_FAST185_STEPS', '3')


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    outp = ROOT / 'out' / LABEL
    if rank == 0:
        outp.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(local_rank)
    sys.argv = ['benchmark_actual_attention126.py',
                '--checkpoint', '/home/mseco/models/MiniMax-H3/FL2VA/transformer',
                '--mode', 'baseline', '--label', LABEL, '--tp', '2', '--usp', '4',
                '--offload', 'native', '--audit', 'strict', '--audit-sequence', 'strict',
                '--steps', STEPS, '--warmup', '1', '--repeats', '1']
    import benchmark_actual_attention126 as bench
    bench.ROOT = ROOT
    bench.main()
    receipt = ROOT / LABEL / f'candidate_rank{rank}.json'
    if receipt.exists() and rank == 0:
        dst = ROOT / 'out' / LABEL / f'rank{rank}.json'
        dst.write_text(receipt.read_text())


if __name__ == '__main__':
    main()
