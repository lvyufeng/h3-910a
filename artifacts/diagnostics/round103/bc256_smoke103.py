#!/usr/bin/env python3
"""MB32/BC256 actual smoke on true192/pad256 keys: poison-padding invariance + CPU Half MATH."""
from pathlib import Path
import json
import os
import time
import statistics
ROOT=Path(os.environ.get('R103_ROOT','/tmp/h3_round103_smoke01'))


def main():
    import torch
    import torch_npu
    from torch.nn.attention import sdpa_kernel,SDPBackend
    import flash_vision103 as kernel
    import best_text87 as previous
    from runtime5gm83 import Runtime5GM
    torch.set_num_threads(2);torch.npu.set_device(0)
    r={'scope':__doc__,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'no_model_weight_or_FloatLinear_teacher':True,'not_full_encoder_timing':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    try:
        meta=kernel.build_kernel(batch=2,m=64,n=192,d=72,out_dir=ROOT/'kernels');r['meta']=meta
        runtime=Runtime5GM(0);aux=torch.empty((16,),dtype=torch.float32,device='npu:0')
        for case in('random','constant_uniform'):
            torch.manual_seed(101)
            q=torch.randn((1,2,64,72),dtype=torch.float16);k=torch.randn((1,2,192,72),dtype=torch.float16);v=torch.randn_like(k)
            if case=='constant_uniform':
                q.zero_();k.zero_();v[0,0].fill_(0.25);v[0,1].fill_(-0.5)
            with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
                gold=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
                gold2=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False)
            assert previous.compare(gold2,gold)['raw_bits_equal']
            kernel.check_host_protocol(q_absmax=float(q.abs().max()),k_absmax=float(k.abs().max()))
            qp=torch.zeros((2,64,80),dtype=torch.float16);qp[...,:72]=q[0]
            packed_q=qp.reshape(2,64,5,16).permute(0,2,1,3).contiguous().to('npu:0')
            outputs={};c={'original_CPU_Half_MATH':previous.tensor_info(gold),'padding':{}}
            for pad in('zero','poison'):
                kp=torch.zeros((2,256,80),dtype=torch.float16);kp[:,:192,:72]=k[0]
                vp=torch.zeros_like(kp);vp[:,:192,:72]=v[0]
                if pad=='poison':
                    kp[:,192:,:72].fill_(20);vp[:,192:,:72].fill_(-23)
                packed_k=kp.reshape(2,256,5,16).permute(0,2,1,3).contiguous().to('npu:0')
                packed_v=vp.reshape(2,16,16,80).permute(0,1,3,2).contiguous().to('npu:0')
                output=torch.empty((2,5,64,16),dtype=torch.float16,device='npu:0')
                buffers=[packed_q,packed_k,packed_v,aux,output]
                before=[previous.tensor_info(x)['sha256']for x in buffers[:3]]
                runtime.launch(meta,buffers)
                unpack=lambda value:value.permute(0,2,1,3).reshape(2,64,80)[...,:72].contiguous().unsqueeze(0)
                first=unpack(output);assert first.shape==gold.shape==(1,2,64,72)
                metric=previous.compare(first,gold);assert metric['finite']
                saved=first.detach().cpu().contiguous();outputs[pad]=saved
                repeat=torch.empty_like(output);buffers[-1]=repeat;runtime.launch(meta,buffers)
                assert previous.compare(unpack(repeat),saved)['raw_bits_equal']
                if case=='constant_uniform':assert metric['raw_bits_equal']
                clocks=[]
                for _ in range(5):
                    torch.npu.synchronize();start=time.perf_counter();runtime.launch(meta,buffers);torch.npu.synchronize()
                    clocks.append(time.perf_counter()-start);assert previous.compare(unpack(repeat),saved)['raw_bits_equal']
                assert before==[previous.tensor_info(x)['sha256']for x in buffers[:3]]
                c['padding'][pad]={'vs_CPU_Half_MATH':metric,'all_repeats_and_inputs_unchanged':True,'clock_s':clocks,'median_s':statistics.median(clocks)}
            c['poison_vs_zero_pad']=previous.compare(outputs['poison'],outputs['zero'])
            assert c['poison_vs_zero_pad']['raw_bits_equal']
            r['cases'][case]=c
        r['runtime']=runtime.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        with(ROOT/'bc256_smoke103.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'cases':r['cases'],'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
