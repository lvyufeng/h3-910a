#!/usr/bin/env python3
"""R143 driver: pinned replay in baseline mode with staging side channel only."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path('/tmp/h3_staging143');LABEL=os.environ.get('H3_STAGING143_LABEL','staging143a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch,torch_npu
    import staging_probe143 as probe
    torch.npu.set_device(local_rank)
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','baseline','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','strict','--steps','25','--warmup','2','--repeats','2']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    original_install=bench._install_layerwise_hook_group if hasattr(bench,'_install_layerwise_hook_group') else None
    # install the probe immediately after model construction by hooking the harness helper
    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    real_group=lb._install_layerwise_hook_group
    def patched_group(blocks,device,stream,pin_memory=True):
        hooks=real_group(blocks,device,stream,pin_memory)
        probe.install(torch,blocks)
        return hooks
    lb._install_layerwise_hook_group=patched_group
    bench.main()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['staging143']=probe.collect(torch)
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
