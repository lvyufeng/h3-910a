#!/usr/bin/env python3
"""R149 driver: pinned replay with a complete device-event step budget."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path('/tmp/h3_budget149');LABEL=os.environ.get('H3_BUDGET149_LABEL','budget149a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch,torch_npu
    import budget_probe149 as probe
    torch.npu.set_device(local_rank)
    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    real_group=lb._install_layerwise_hook_group
    def patched(blocks,device,stream,pin_memory=True):
        hooks=real_group(blocks,device,stream,pin_memory)
        probe.install(torch,blocks)
        return hooks
    lb._install_layerwise_hook_group=patched
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','baseline','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','strict','--steps','25','--warmup','2','--repeats','1']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    bench.main()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['budget149']=probe.collect(torch)
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
