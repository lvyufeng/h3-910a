#!/usr/bin/env python3
"""R186 runner: one 8-rank replay in verify mode (native and fast verdicts compared per layer)."""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path('/tmp/h3_verify186')
LABEL = os.environ['H3_VERIFY186_LABEL']


def main():
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(local_rank)
    sys.argv = ['benchmark_actual_attention126.py',
                '--checkpoint', '/home/mseco/models/MiniMax-H3/FL2VA/transformer',
                '--mode', 'baseline', '--label', LABEL, '--tp', '2', '--usp', '4',
                '--offload', 'native', '--audit', 'strict', '--audit-sequence', 'strict',
                '--steps', os.environ.get('H3_VERIFY186_STEPS', '10'),
                '--warmup', '1', '--repeats', '1']
    import benchmark_actual_attention126 as bench
    bench.ROOT = ROOT
    bench.main()


if __name__ == '__main__':
    main()
