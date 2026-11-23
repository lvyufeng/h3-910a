#!/usr/bin/env python3
"""R140 roofline probe driver: run the pinned R126 replay in profile mode.

Nothing here changes math: the replay harness asserts bit-identical predictions
against the R124 baseline, and this driver only adds side-channel timing.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys

ROOT=Path('/tmp/h3_roofline140')
LABEL=os.environ.get('H3_ROOFLINE140_LABEL','roofline140a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch
    import torch_npu
    import roofline_probe140 as probe
    torch.npu.set_device(local_rank)
    device=torch.device(f'npu:{local_rank}')
    probe.install_prefetch_timing(torch)
    bandwidth=probe.h2d_bandwidth(torch,device)
    sys.argv=['benchmark_actual_attention126.py','--checkpoint','/tmp/h3_align12401/original_metadata124f/FL2VA',
        '--mode','profile','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','profile','--steps','25','--warmup','2','--repeats','2']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    bench.main()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['roofline140']={'prefetch':probe.collect_prefetch_timing(torch),'h2d_bandwidth':bandwidth,
            'measurement_only_no_math_change':True,'region_timing_uses_synchronize_coarse_attribution':True,
            'parent_child_regions_inclusive_do_not_sum':True}
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
