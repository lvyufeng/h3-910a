#!/usr/bin/env python3
"""Probe portable FP32-UB bias plus one Half cast against original CPU Half Linear.

No model teacher is changed. Timings here are micro packed-kernel timings,
not full H3 encoder performance. The root stages and runs this source later.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def tensor_sha(value):
    import torch
    data=value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def compare(value,reference):
    import torch
    a=value.detach().cpu().contiguous();b=reference.detach().cpu().contiguous()
    assert a.shape==b.shape and a.dtype==b.dtype
    error=a.double()-b.double()
    return {'dtype':str(a.dtype),'shape':list(a.shape),'finite':bool(torch.isfinite(a).all()and torch.isfinite(b).all()),
            'raw_bits_equal':bool(torch.equal(a.view(torch.uint8),b.view(torch.uint8))),
            'max_abs_error':float(error.abs().max()),'RMSE':float(error.square().mean().sqrt()),
            'relative_RMS':float(error.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-30)),
            'sha256':tensor_sha(a),'reference_sha256':tensor_sha(b)}


def main():
    import argparse
    import numpy as np
    import torch
    import torch_npu
    from runtime5gm83 import Runtime5GM
    import tik_bias_epilogue83 as kernel
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='/tmp/h3_round83_bias01')
    args=parser.parse_args();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'bias_result83.json'
    if receipt_path.exists():raise RuntimeError('Refusing to replay or overwrite bias evidence')
    r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'kernel_source_sha256':hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest(),
       'geometry':{'M':128,'K':512,'N':128},'kernel_launch_executed':False,
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'not_H3_performance':True}
    try:
        torch.set_num_threads(4);m,k,n=128,512,128
        rng=np.random.default_rng(82082)
        ac=torch.from_numpy(rng.standard_normal((m,k)).astype(np.float16))
        wc=torch.from_numpy(rng.standard_normal((n,k)).astype(np.float16))
        bc=torch.from_numpy((rng.standard_normal(n)*.1).astype(np.float16))
        before={'x':tensor_sha(ac),'weight':tensor_sha(wc),'bias':tensor_sha(bc)}
        with torch.inference_mode():
            cpu_half=torch.nn.functional.linear(ac,wc,bc)
            cpu_repeat=torch.nn.functional.linear(ac,wc,bc)
            FP32_epilogue_diagnostic=torch.nn.functional.linear(ac.float(),wc.float(),bc.float()).half()
        if not compare(cpu_repeat,cpu_half)['raw_bits_equal']:raise RuntimeError('CPU Half reference is not bit-stable')
        r['CPU_reference_operator']='original torch.nn.functional.linear(Half input,Half weight,Half bias)'
        r['FP32_diagnostic_not_a_new_teacher']=True
        meta=kernel.build_kernel(variant='fp32_bias_fp16_out',m=m,k=k,n=n,out_dir=root/'kernels')
        r['build']=meta
        if meta['dst_dtype']!='float16':raise RuntimeError('Portable bias kernel is not a one-Half-cast output')
        runtime=Runtime5GM(0);dev=torch.device('npu',0)
        a=ac.reshape(m,k//16,16).permute(1,0,2).contiguous().to(dev)
        w=wc.reshape(n,k//16,16).permute(1,0,2).contiguous().to(dev)
        bias=bc.float().to(dev);aux=torch.zeros((16,),dtype=torch.float16,device=dev)
        out=torch.empty((n//16,m,16),dtype=torch.float16,device=dev)
        a_nd=ac.to(dev);w_nd=wc.to(dev);bias_half=bc.to(dev)
        if not compare(bias.cpu().half(),bc)['raw_bits_equal']:raise RuntimeError('Kernel bias values are not the original Half parameters')
        buffers=[a,w,bias,aux,out];runtime.register(meta)
        gpu_before={name:tensor_sha(value)for name,value in zip(('a','weight','bias','aux'),buffers[:-1])}
        def execute():
            runtime.launch(meta,buffers)
            return out.permute(1,0,2).contiguous().reshape(m,n).cpu()
        with torch.inference_mode():
            first=execute();r['kernel_launch_executed']=True;repeat=execute()
            native=torch.nn.functional.linear(a_nd,w_nd,bias_half).cpu()
            native_order=(torch.nn.functional.linear(a_nd,w_nd)+bias_half).cpu()
        r['kernel_vs_CPU_FP16']=compare(first,cpu_half)
        r['native_vs_CPU_FP16']=compare(native,cpu_half)
        r['repeat_vs_first']=compare(repeat,first)
        r['kernel_vs_FP32_epilogue_diagnostic']=compare(first,FP32_epilogue_diagnostic)
        r['native_vs_Half_GEMM_then_Half_bias']=compare(native,native_order)
        r['original_inputs_and_parameters_unchanged']=before=={'x':tensor_sha(ac),'weight':tensor_sha(wc),'bias':tensor_sha(bc)}
        if not r['repeat_vs_first']['raw_bits_equal']or not r['kernel_vs_CPU_FP16']['finite']or not r['original_inputs_and_parameters_unchanged']:
            raise RuntimeError('Bias kernel finite/repeat/frozen input guard failed')
        timings=[]
        for _ in range(3):
            torch.npu.synchronize();start=time.perf_counter();runtime.launch(meta,buffers)
            timings.append(time.perf_counter()-start)
        r['packed_kernel_only_s_including_sync']=timings;r['packed_micro_median_s']=statistics.median(timings)
        r['GPU_operands_unchanged']=gpu_before=={name:tensor_sha(value)for name,value in zip(('a','weight','bias','aux'),buffers[:-1])}
        if not r['GPU_operands_unchanged']:raise RuntimeError('Bias kernel mutated a GM operand')
        r['runtime']=runtime.snapshot();r['numerical_acceptance_threshold']=None;r['completed']=True
    except BaseException as error:
        r['failure']={'type':type(error).__name__,'message':str(error)[:4000]};raise
    finally:
        with receipt_path.open('x')as f:f.write(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'receipt':str(receipt_path),'completed':r.get('completed',False)},indent=2))


if __name__=='__main__':main()
