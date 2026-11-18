#!/usr/bin/env python3
"""Direct Float-NZ softmax smoke on true/padded real geometry and HalfPV output."""
from pathlib import Path
import json
import time
import statistics
ROOT=Path('/tmp/h3_round97_nz01')


def main():
    import torch
    import torch_npu
    import softmax_pack97 as kernel
    import pv_high_low96 as pv
    from runtime5gm83 import Runtime5GM
    from full_row_fp32_fixed84 import compare,tensor_info,file_sha
    torch.set_num_threads(2);torch.npu.set_device(0);runtime=Runtime5GM(0)
    r={'scope':__doc__,'completed':False,'source_SHA256':file_sha(Path(__file__)),
       'no_FP32Linear_teacher_or_BF_goldcasts':True,'not_full_encoder_timing':True,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    try:
        for m in(512,448):
            meta=kernel.build_kernel(batch=16,m=m,n=4096,valid_keys=4032,out_dir=ROOT/'softmax')
            torch.manual_seed(m)
            scores=torch.randn((16,m,4096),dtype=torch.float32,device='cpu')*3
            logical=torch.zeros((16,512,4096),dtype=torch.float32);logical[:,:m,:]=scores
            # Padded keys are huge positive: the kernel must exclude them before max/exp.
            logical[:,:,4032:]=500.0
            packed=logical.reshape(16,512,256,16).permute(0,2,1,3).contiguous().to('npu:0')
            gold=torch.softmax(scores[:,:,:4032]*72**-0.5,dim=-1)
            hi=torch.empty((16,256,512,16),dtype=torch.float16,device='npu:0');lo=torch.empty_like(hi)
            aux=torch.empty((16,),dtype=torch.float32,device='npu:0')
            before=tensor_info(packed)['sha256']
            runtime.launch(meta,[packed,aux,aux,hi,lo])
            unpack=lambda x:pv.unpack_out_half(x,512)
            hiout,loout=unpack(hi),unpack(lo)
            recovered=hiout[:,:m,:4032].float()+loout[:,:m,:4032].float()/1024
            metrics=compare(recovered.cpu(),gold)
            assert metrics['finite']
            assert torch.count_nonzero(hiout[:,:,4032:]).item()==0 and torch.count_nonzero(loout[:,:,4032:]).item()==0
            if m==448:assert torch.count_nonzero(hiout[:,448:,:]).item()==0 and torch.count_nonzero(loout[:,448:,:]).item()==0
            repeat_hi=torch.empty_like(hi);repeat_lo=torch.empty_like(lo)
            runtime.launch(meta,[packed,aux,aux,repeat_hi,repeat_lo])
            assert compare(repeat_hi,hi)['raw_bits_equal']and compare(repeat_lo,lo)['raw_bits_equal']
            clocks=[]
            for _ in range(5):
                torch.npu.synchronize();start=time.perf_counter();runtime.launch(meta,[packed,aux,aux,hi,lo]);torch.npu.synchronize();clocks.append(time.perf_counter()-start)
            assert before==tensor_info(packed)['sha256']
            row_sums=recovered.sum(-1)
            r['cases'][str(m)]={'meta':meta,'recovered_float_probabilities_vs_CPU':metrics,
                               'sum_max_abs_error':float((row_sums-1).abs().max()),
                               'key_tail64_zero_and_padded_M_rows_zero':True,'repeat_bits_equal':True,'input_SHA_unchanged':True,
                               'runtime_s':clocks,'median_s':statistics.median(clocks)}
        r['completed']=True
    except BaseException as e:r['failure']={'type':type(e).__name__,'message':str(e)};raise
    finally:
        with(ROOT/'softmax_smoke97.json').open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'completed':r['completed'],'cases':r['cases'],'failure':r.get('failure')},indent=2),flush=True)


if __name__=='__main__':main()
