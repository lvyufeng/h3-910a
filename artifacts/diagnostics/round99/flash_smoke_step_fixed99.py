#!/usr/bin/env python3
"""Small actual original-910 mixed-Cube/Float-online flash smoke, no model speed claim."""
from pathlib import Path
import json
import os
import time
import statistics
ROOT=Path(os.environ.get('R99_ROOT','/tmp/h3_round99_smoke02'))


def main():
    import torch
    import torch_npu
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import flash_vision99 as kernel
    import best_text87 as previous
    from runtime5gm83 import Runtime5GM
    torch.set_num_threads(2);torch.npu.set_device(0)
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'no_model_weight_or_FloatLinear_teacher':True,'not_full_encoder_timing':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None}
    try:
        torch.manual_seed(99)
        q,k,v=[torch.randn((1,2,128,72),dtype=torch.float16)for _ in range(3)]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
            gold=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
            gold_repeat=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
        assert previous.compare(gold_repeat,gold)['raw_bits_equal']
        r['bound']=kernel.check_host_protocol(q_absmax=float(q.abs().max()),k_absmax=float(k.abs().max()))
        inputs=[]
        for original in(q,k):
            padded=torch.zeros((2,128,80),dtype=torch.float16);padded[...,:72]=original[0]
            inputs.append(padded.reshape(2,128,5,16).permute(0,2,1,3).contiguous().to('npu:0'))
        padded=torch.zeros((2,128,80),dtype=torch.float16);padded[...,:72]=v[0]
        packed_v=padded.reshape(2,8,16,80).permute(0,1,3,2).contiguous().to('npu:0')
        aux=torch.empty((16,),dtype=torch.float32,device='npu:0')
        output=torch.empty((2,5,128,16),dtype=torch.float16,device='npu:0')
        meta=kernel.build_kernel(batch=2,m=128,n=128,d=72,out_dir=ROOT/'kernels')
        r['meta']=meta;runtime=Runtime5GM(0)
        before=[previous.tensor_info(x)['sha256']for x in inputs+[packed_v]]
        runtime.launch(meta,inputs+[packed_v,aux,output])
        unpack=lambda x:x.permute(0,2,1,3).reshape(2,128,80)[...,:72].contiguous().unsqueeze(0)
        first=unpack(output);assert first.shape==gold.shape==(1,2,128,72)
        r['vs_original_CPU_Half_MATH']=previous.compare(first,gold);assert r['vs_original_CPU_Half_MATH']['finite']
        repeat=torch.empty_like(output);runtime.launch(meta,inputs+[packed_v,aux,repeat])
        r['repeat']=previous.compare(unpack(repeat),first);assert r['repeat']['raw_bits_equal']
        clocks=[]
        for _ in range(5):
            torch.npu.synchronize();start=time.perf_counter();runtime.launch(meta,inputs+[packed_v,aux,repeat]);torch.npu.synchronize()
            clocks.append(time.perf_counter()-start);assert previous.compare(unpack(repeat),first)['raw_bits_equal']
        assert before==[previous.tensor_info(x)['sha256']for x in inputs+[packed_v]]
        r['inputs_unchanged']=True;r['clock_s']=clocks;r['median_s']=statistics.median(clocks)
        r['runtime']=runtime.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'flash_smoke99.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'vs_CPU':r.get('vs_original_CPU_Half_MATH'),'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
