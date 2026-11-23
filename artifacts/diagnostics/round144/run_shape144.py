#!/usr/bin/env python3
"""R144 driver: pinned replay with a shape side channel."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
ROOT=Path('/tmp/h3_shapes144');LABEL=os.environ.get('H3_SHAPES144_LABEL','shapes144a')

def main():
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    out=ROOT/LABEL
    assert out.is_dir(),'launcher must pre-create the output directory'
    sys.path.insert(0,str(ROOT))
    import torch,torch_npu
    import shape_probe144 as probe
    torch.npu.set_device(local_rank)
    import vllm_omni.diffusion.offloader.layerwise_backend as lb
    real_group=lb._install_layerwise_hook_group
    def patched(blocks,device,stream,pin_memory=True):
        hooks=real_group(blocks,device,stream,pin_memory)
        return hooks
    lb._install_layerwise_hook_group=patched
    sys.argv=['benchmark_actual_attention126.py',
        '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
        '--mode','baseline','--label',LABEL,'--tp','2','--usp','4','--offload','native',
        '--audit-sequence','strict','--steps','25','--warmup','1','--repeats','1']
    import benchmark_actual_attention126 as bench
    bench.ROOT=ROOT
    real_main=bench.main
    def main_with_probe():
        # install right before the model runs: the harness builds the model inside main,
        # so wrap MiniMaxH3DiTModel construction to grab the instance.
        import h3_replay123.transformer as replay
        original_cls=replay.MiniMaxH3DiTModel
        class Wrapped(original_cls):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                probe.install(torch,self)
        replay.MiniMaxH3DiTModel=Wrapped
        return real_main()
    main_with_probe()
    receipt=out/f'candidate_rank{rank}.json'
    if receipt.exists():
        data=json.loads(receipt.read_text())
        data['shapes144']=probe.collect()
        receipt.write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
