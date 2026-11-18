#!/usr/bin/env python3
"""Probe blocked HalfA/HalfB->FP32-output GEMM at H3 matrix sizes.

Actual CPU Half Linear remains the numerical reference;FP32 Linear is only an
accumulator diagnostic. Cached weight packing is reported separately. Resident
GPU end-to-end timings include A packing and FP32-output unpacking,not model
startup and not the whole encoder. No Half narrowing occurs inside the kernel.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time
from probe_bias83 import compare,tensor_sha

CASES={'smoke':(128,256,128),'kv':(2043,5120,128),'q':(2043,5120,1024),'up':(2043,5120,3456),'row':(2043,3456,5120)}


def main():
    import argparse
    import torch
    import torch_npu
    from runtime5gm83 import Runtime5GM
    import blocked_fp32_gemm83 as kernel
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=sorted(CASES),default='kv')
    parser.add_argument('--root',default='/tmp/h3_round83_blocked01');args=parser.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    result_path=root/('blocked_'+args.case+'_result83.json')
    if result_path.exists():raise RuntimeError('Refusing to replay or overwrite blocked GEMM evidence')
    m,k,n=CASES[args.case];mp=((m+127)//128)*128
    r={'scope':__doc__,'case':args.case,'logical_geometry':{'M':m,'K':k,'N':n},'padded_M':mp,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'kernel_source_sha256':hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
       'kernel_launch_executed':False,'full_encoder_precision_accepted':False,'whole_goal_complete':False,
       'not_whole_encoder_timing':True,'precision_acceptance_threshold_chosen':None,
       'predeclared_wrong_layout_guard_RMS':0.01}
    try:
        torch.set_num_threads(8);torch.manual_seed(830830+list(CASES).index(args.case))
        ac=(torch.randn((m,k))*.5).half();wc=(torch.randn((n,k))*.05).half()
        before={'input':tensor_sha(ac),'weight':tensor_sha(wc)}
        start=time.perf_counter()
        with torch.inference_mode():
            cpu_half=torch.nn.functional.linear(ac,wc)
            cpu_repeat=torch.nn.functional.linear(ac,wc)
            fp32_diagnostic=torch.nn.functional.linear(ac.float(),wc.float())
        r['excluded_CPU_reference_s']=time.perf_counter()-start
        if not compare(cpu_half,cpu_repeat)['raw_bits_equal']:raise RuntimeError('CPU Half operator is not bit-stable')
        r['CPU_reference_operator']='torch.nn.functional.linear(Half input,Half weight),no bias'
        r['FP32_CPU_is_accumulator_diagnostic_not_teacher']=True
        start=time.perf_counter();meta=kernel.build_kernel(m=m,k=k,n=n,out_dir=root/'kernels')
        r['build']=meta;r['excluded_build_s']=time.perf_counter()-start
        if meta['dst_dtype']!='float32':raise RuntimeError('Blocked partial GEMM unexpectedly narrows to Half')
        runtime=Runtime5GM(0);dev=torch.device('npu',0);a_nd=ac.to(dev);w_nd=wc.to(dev)
        with torch.inference_mode():
            start=time.perf_counter()
            wpacked=w_nd.reshape(n,k//16,16).permute(1,0,2).contiguous()
            torch.npu.synchronize();r['excluded_weight_pack_s']=time.perf_counter()-start
            r['cached_Half_weight_bytes']=wpacked.numel()*wpacked.element_size()
            if not compare(wpacked.permute(1,0,2).contiguous().reshape(n,k).cpu(),wc)['raw_bits_equal']:
                raise RuntimeError('Packed weight values differ from original Half parameters')
            aux=[torch.zeros((16,),dtype=torch.float16,device=dev)for _ in range(2)]
            out=torch.empty((n//16,mp,16),dtype=torch.float32,device=dev)
            def pack_a():
                padded=torch.zeros((mp,k),dtype=torch.float16,device=dev);padded[:m].copy_(a_nd)
                return padded.reshape(mp,k//16,16).permute(1,0,2).contiguous()
            apacked=pack_a()
            if not compare(apacked.permute(1,0,2).contiguous().reshape(mp,k)[:m].cpu(),ac)['raw_bits_equal']:
                raise RuntimeError('Packed input values differ from original Half input')
            if int(torch.count_nonzero(apacked.permute(1,0,2).contiguous().reshape(mp,k)[m:]))!=0:
                raise RuntimeError('Padding rows are not exactly zero')
            runtime.register(meta)
            def unpack():return out.permute(1,0,2).contiguous().reshape(mp,n)
            buffers=[apacked,wpacked,*aux,out]
            runtime.launch(meta,buffers);r['kernel_launch_executed']=True
            first=unpack()[:m].cpu();padded_output=unpack()[m:]
            r['padded_output_rows_max_abs']=float(padded_output.abs().max())if padded_output.numel()else 0.0
            runtime.launch(meta,buffers);repeat=unpack()[:m].cpu()
            native=torch.nn.functional.linear(a_nd,w_nd).cpu()
            native_repeat=torch.nn.functional.linear(a_nd,w_nd).cpu()
        r['kernel_FP32_vs_accumulator_diagnostic']=compare(first,fp32_diagnostic)
        r['kernel_final_Half_vs_CPU_Half']=compare(first.half(),cpu_half)
        r['native_Half_vs_CPU_Half']=compare(native,cpu_half)
        r['kernel_repeat_vs_first']=compare(repeat,first)
        r['native_repeat_vs_first']=compare(native_repeat,native)
        r['original_inputs_and_weights_unchanged']=before=={'input':tensor_sha(ac),'weight':tensor_sha(wc)}
        if (not r['kernel_repeat_vs_first']['raw_bits_equal']or not r['kernel_FP32_vs_accumulator_diagnostic']['finite']
                or r['padded_output_rows_max_abs']!=0 or not r['original_inputs_and_weights_unchanged']):
            raise RuntimeError('Blocked GEMM repeat/finite/padding/frozen input guard failed')
        if r['kernel_FP32_vs_accumulator_diagnostic']['relative_RMS']>r['predeclared_wrong_layout_guard_RMS']:
            raise RuntimeError('Blocked GEMM failed the predeclared wrong-layout sanity guard;no timing claim')
        packed_times=[];end_to_end=[];native_times=[]
        for _ in range(3):
            with torch.inference_mode():
                torch.npu.synchronize();start=time.perf_counter();runtime.launch(meta,buffers)
                packed_times.append(time.perf_counter()-start)
                torch.npu.synchronize();start=time.perf_counter()
                fresh_a=pack_a();runtime.launch(meta,[fresh_a,wpacked,*aux,out],synchronize=False)
                result=unpack();torch.npu.synchronize();end_to_end.append(time.perf_counter()-start)
                torch.npu.synchronize();start=time.perf_counter()
                native_value=torch.nn.functional.linear(a_nd,w_nd);torch.npu.synchronize()
                native_times.append(time.perf_counter()-start)
        r['packed_kernel_only_s']=packed_times;r['resident_end_to_end_with_A_pack_and_output_unpack_s']=end_to_end
        r['resident_native_Half_Linear_s']=native_times
        r['medians_s']={'packed_kernel':statistics.median(packed_times),'resident_with_layouts':statistics.median(end_to_end),
                        'native_Half':statistics.median(native_times)}
        r['native_returns_Half_while_custom_retains_FP32']=True;r['runtime']=runtime.snapshot();r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)[:4000]};raise
    finally:
        with result_path.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'receipt':str(result_path),'completed':r.get('completed',False)},indent=2))


if __name__=='__main__':main()
