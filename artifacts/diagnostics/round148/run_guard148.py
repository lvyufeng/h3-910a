#!/usr/bin/env python3
"""R148 driver: pinned replay that compares the guard fast path against the native path."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path('/tmp/h3_guard148');LABEL=os.environ.get('H3_GUARD148_LABEL','guard148a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch,torch_npu
    import guard_probe148 as probe
    torch.npu.set_device(local_rank)
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','baseline','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','strict','--steps','25','--warmup','1','--repeats','1']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    probe.install(torch)
    bench.main()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['guard148']=probe.collect()
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
