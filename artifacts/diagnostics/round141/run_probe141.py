#!/usr/bin/env python3
"""R141 driver: pinned R126 replay in profile mode plus collective/staging side channel."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT=Path('/tmp/h3_roofline141')
LABEL=os.environ.get('H3_ROOFLINE141_LABEL','roofline141a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch
    import torch_npu
    import roofline_probe141 as probe
    torch.npu.set_device(local_rank)
    device=torch.device(f'npu:{local_rank}')
    probe.install_tp_reduce_timing(torch)
    probe.install_prefetch_timing(torch)
    bandwidth=probe.h2d_bandwidth(torch,device)
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','profile','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','profile','--steps','25','--warmup','2','--repeats','2']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    bench.main()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['roofline141']=probe.collect(torch)
        data['roofline141']['h2d_bandwidth']=bandwidth
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
