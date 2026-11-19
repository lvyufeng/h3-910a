#!/usr/bin/env python3
"""Actual TP8 Half-column/Float-row GEMMs with HCCL and original CPU Half Linear controls."""
from pathlib import Path
import json
import os
import statistics
import time
ROOT=Path(os.environ.get('R107_ROOT','/tmp/h3_round107_gemm_tp01'))


def main():
    import torch
    import torch_npu
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
    from runtime5gm83 import Runtime5GM
    import best_text87 as previous
    from vision_gemm_backend107 import VisionGemmBackend107
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'CPU_reference_Linears_inputs_weights_bias_output_all_Half':True,'no_FloatLinear_teacher_or_hidden_conditioning':True,
       'not_full_encoder_timing':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{}}
    group=None;ready=False
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank)
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        backend=VisionGemmBackend107(Runtime5GM(rank),ROOT/'tp_backend_kernels')
        for case,mode,fullk,fulln in(('qkv','column',1152,3456),('out_proj','row',1152,1152),('fc1','column',1152,4304),('fc2','row',4304,1152)):
            torch.manual_seed(107)
            x=torch.randn((128,fullk),dtype=torch.float16);w=torch.randn((fulln,fullk),dtype=torch.float16)*0.03125
            bias=torch.randn((fulln,),dtype=torch.float16)*0.03125
            with torch.inference_mode():gold=torch.nn.functional.linear(x,w,bias);gold2=torch.nn.functional.linear(x,w,bias)
            assert gold.dtype==torch.float16 and previous.compare(gold2,gold)['raw_bits_equal']
            if mode=='column':
                if case=='qkv':
                    indices=torch.cat([torch.arange(s*1152+rank*144,s*1152+(rank+1)*144)for s in range(3)])
                else:indices=torch.arange(rank*538,(rank+1)*538)
                localw=w.index_select(0,indices).contiguous().to(torch.device('npu',rank));localb=bias.index_select(0,indices).contiguous().to(localw.device)
                localx=x.to(localw.device);localgold=gold.index_select(1,indices).contiguous()
            else:
                width=144 if case=='out_proj'else 538;start=rank*width
                localw=w[:,start:start+width].contiguous().to(torch.device('npu',rank));localb=None
                localx=x[:,start:start+width].contiguous().to(localw.device);localgold=gold
            entry=backend.prime(case,localw,localb,mode,m=128)
            controls=backend.controls(entry,localx)
            original=[previous.tensor_info(t)['sha256']for t in(localw,localx)]
            fullbias=bias.to(localw.device).float()if mode=='row'else None
            calls={'Float_collectives':0,'Half_output_casts_after_collective':0}
            def run():
                partial=backend.run(entry,localx)
                if mode=='column':assert partial.dtype==torch.float16;return partial
                assert partial.dtype==torch.float32
                summed=group.all_reduce(partial);assert summed.dtype==torch.float32 and summed.shape==partial.shape
                calls['Float_collectives']+=1
                out=(summed+fullbias).half();calls['Half_output_casts_after_collective']+=1
                return out
            with torch.inference_mode():first=run();repeated=run()
            assert first.shape==localgold.shape and previous.compare(repeated,first)['raw_bits_equal']
            clocks=[]
            for _ in range(5):
                torch.distributed.barrier();torch.npu.synchronize();begin=time.perf_counter()
                with torch.inference_mode():out=run()
                torch.npu.synchronize();clocks.append(time.perf_counter()-begin)
                assert previous.compare(out,first)['raw_bits_equal']
            assert original==[previous.tensor_info(t)['sha256']for t in(localw,localx)]
            c={'mode':mode,'local_geometry':{key:entry[key]for key in('m','k','n','mp','kp','np')},'kernel':entry['meta'],
               'controls':controls,'vs_CPU_Half_Linear':previous.compare(first,localgold),'all_repeats_and_original_inputs_weights_unchanged':True,
               'clock_s':clocks,'median_s':statistics.median(clocks),'counters':calls}
            if mode=='row':
                allhash=[None]*8;torch.distributed.all_gather_object(allhash,c['vs_CPU_Half_Linear']['npu']['sha256'],group=group.cpu_group)
                assert len(set(allhash))==1;c['all8_output_SHA_equal']=True
            r['cases'][case]=c
        r['backend']=backend.snapshot();r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if group is not None:group.destroy()
        if ready:destroy_model_parallel();torch.distributed.destroy_process_group()
        with(ROOT/('gemm_tp_rank%d.json'%rank)).open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'rank':rank,'completed':r['completed'],'cases':list(r['cases']),'failure':r.get('failure')}),flush=True)


if __name__=='__main__':main()
