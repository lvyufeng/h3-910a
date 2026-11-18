#!/usr/bin/env python3
"""Compile six vision kernels and verify a two-K-block biased Half smoke."""
from pathlib import Path
import json
ROOT=Path('/tmp/h3_round87_vision01')


def main():
    import torch
    import torch_npu
    import vision_bias87 as kernel
    from runtime5gm83 import Runtime5GM
    from full_row_fp32_fixed84 import compare,tensor_info
    from vision_adapter87 import SHAPES
    torch.npu.set_device(0);torch.set_num_threads(2);runtime=Runtime5GM(0)
    torch.manual_seed(8701)
    a=torch.randn((128,256),dtype=torch.float16)*.125
    w=torch.randn((128,256),dtype=torch.float16)*.125
    b=torch.randn((128,),dtype=torch.float16)*.125
    gold=torch.nn.functional.linear(a,w,b)
    meta=kernel.build_kernel(m=128,k=256,n=128,out_dir=ROOT/'kernels')
    pa=a.to('npu:0').reshape(128,16,16).permute(1,0,2).contiguous()
    pw=w.to('npu:0').reshape(128,16,16).permute(1,0,2).contiguous()
    pb=b.to('npu:0').float();aux=torch.zeros(16,dtype=torch.float16,device='npu:0')
    out=torch.empty((8,128,16),dtype=torch.float16,device='npu:0')
    runtime.launch(meta,[pa,pw,pb,aux,out],synchronize=True)
    first=out.permute(1,0,2).reshape(128,128).cpu()
    runtime.launch(meta,[pa,pw,pb,aux,out],synchronize=True)
    repeat=compare(out.permute(1,0,2).reshape(128,128),first);assert repeat['raw_bits_equal']
    result=compare(first,gold)
    assert result['finite']and result['relative_rms']<.01 # predeclared wrong-layout sanity,not model tolerance
    records=[kernel.build_kernel(m=m,k=k,n=n,out_dir=ROOT/'kernels')for m,k,n in sorted(set(SHAPES.values()))]
    receipt={'scope':__doc__,'small_biased_CPU_Half_compare':result,'repeat_bits':repeat,
             'six_actual_padded_shapes_prebuilt':records,'no_full_encoder_precision_claim':True}
    target=ROOT/'prebuild_vision87.json';assert not target.exists()
    with target.open('x')as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print('BiasedtwoKblocksmokelaunched/repeated;all6actualvisionkernelscompiled')


if __name__=='__main__':main()
