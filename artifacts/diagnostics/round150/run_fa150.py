#!/usr/bin/env python3
"""R150 driver: baseline mode plus the R126 event-level attention profiler.

--mode baseline keeps the harness from installing its synchronising region hooks,
while --audit-sequence profile installs ActualAttentionProfile126, which records
device events for the FA wrapper/padding/core/ACL launch, USP pre/post and q/k norm
and asserts on every call that no host synchronisation was added.
"""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path('/tmp/h3_fa150');LABEL=os.environ.get('H3_FA150_LABEL','fa150a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch,torch_npu
    torch.npu.set_device(local_rank)
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','baseline','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','profile','--steps','25','--warmup','2','--repeats','2']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    bench.main()

if __name__=='__main__':main()
