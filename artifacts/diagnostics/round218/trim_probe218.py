#!/usr/bin/env python3
"""Bounded single-card FA experiment with fixed inputs and fail-closed results.

No weights/activation archives leave ts-217. Each variant runs in its own process and cache.
Reports clean wrapper wall time, wrapper device time and ACL-launch kernel events separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback


def digest(tensor):
    import torch
    host=tensor.detach().cpu().contiguous()
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--n',type=int,required=True)
    parser.add_argument('--heads',type=int,default=7)
    parser.add_argument('--pad',type=int,default=23)
    parser.add_argument('--reps',type=int,default=7)
    args=parser.parse_args()
    root=Path('/tmp/h3_trim218')
    output=root/'out'/args.label
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    report={'label':args.label,'input_shape':[1,args.heads,args.n,128],
            'original_pad':args.pad,'valid_keys':args.n-args.pad,
            'seed':20261124,'model_weights_loaded':False,'production_changed':False,
            'complete':False,'scope':'single-card local FA diagnostic, NOT whole TP2+USP4 step'}
    try:
        import torch
        import torch_npu
        import fa218trim as fa
        torch.npu.set_device(0)
        shape=tuple(report['input_shape'])
        generator=torch.Generator(device='cpu').manual_seed(report['seed'])
        inputs=tuple(torch.randn(shape,dtype=torch.float16,generator=generator).to('npu:0') for _ in range(3))
        torch.npu.synchronize()
        report['input_sha256']=[digest(t) for t in inputs]
        # Build/cache prime is not part of warm timing.
        begin=time.perf_counter()
        value=fa.flash_attention_910a(*inputs,n_pad=args.pad)
        torch.npu.synchronize()
        report['cold_build_and_first_call_s']=time.perf_counter()-begin
        report['output_sha256']=digest(value)
        report['input_sha256_after_prime']=[digest(t) for t in inputs]
        # Exact word counts do not rely on defective FP16 isnan/isfinite kernels.
        words=value.detach().cpu().contiguous().view(torch.int16).numpy().astype('uint16')
        magnitude=words & 0x7fff
        report['output_nonfinite_count']=int((magnitude>=0x7c00).sum())
        report['output_saturation_count']=int((magnitude==0x7bff).sum())
        report['output_numel']=value.numel()
        for _ in range(3):
            fa.flash_attention_910a(*inputs,n_pad=args.pad)
        torch.npu.synchronize()
        times=[]
        for _ in range(args.reps):
            torch.npu.synchronize()
            begin=time.perf_counter()
            fa.flash_attention_910a(*inputs,n_pad=args.pad)
            torch.npu.synchronize()
            times.append(1000*(time.perf_counter()-begin))
        report['warm_wrapper_wall_ms']=times
        report['warm_wrapper_wall_median_ms']=statistics.median(times)
        # Event timing is measured separately from the clean timing above.
        class LaunchEvents:
            def __init__(self,lib):
                self.lib=lib; self.pairs=[]
            def __getattr__(self,name):
                return getattr(self.lib,name)
            def fa910a_launch(self,*a):
                first=torch.npu.Event(enable_timing=True)
                last=torch.npu.Event(enable_timing=True)
                first.record()
                rc=self.lib.fa910a_launch(*a)
                last.record()
                self.pairs.append((first,last,rc))
                return rc
        proxy=LaunchEvents(fa._LIB)
        fa._LIB=proxy
        wrapper_events=[]
        for _ in range(args.reps):
            first=torch.npu.Event(enable_timing=True)
            last=torch.npu.Event(enable_timing=True)
            first.record()
            fa.flash_attention_910a(*inputs,n_pad=args.pad)
            last.record()
            wrapper_events.append((first,last))
        torch.npu.synchronize()
        # Keep the queued torch events as raw diagnostic evidence, but use native
        # ACL event records on the exact stream of the direct launch below. With
        # torch task queues, record() can be submitted after a ctypes launch.
        report['torch_kernel_event_ms_raw']=[float(x.elapsed_time(y)) for x,y,rc in proxy.pairs]
        report['wrapper_event_ms']=[float(x.elapsed_time(y)) for x,y in wrapper_events]
        report['wrapper_event_median_ms']=statistics.median(report['wrapper_event_ms'])
        report['instrumented_acl_launches']=len(proxy.pairs)
        assert len(proxy.pairs)==args.reps
        assert all(rc==0 for x,y,rc in proxy.pairs)
        import ctypes
        acl=proxy.lib
        acl.aclrtCreateEventWithFlag.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint32]
        acl.aclrtCreateEventWithFlag.restype=ctypes.c_int
        acl.aclrtRecordEvent.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        acl.aclrtRecordEvent.restype=ctypes.c_int
        acl.aclrtSynchronizeEventWithTimeout.argtypes=[ctypes.c_void_p,ctypes.c_int32]
        acl.aclrtSynchronizeEventWithTimeout.restype=ctypes.c_int
        acl.aclrtEventElapsedTime.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]
        acl.aclrtEventElapsedTime.restype=ctypes.c_int
        acl.aclrtDestroyEvent.argtypes=[ctypes.c_void_p]
        acl.aclrtDestroyEvent.restype=ctypes.c_int
        def check(rc,what):
            if rc: raise RuntimeError(f'{what} rc={rc}')
        # Re-use stable, already-produced NZ inputs and invoke only the ACL kernel.
        scratch=next(iter(fa._SCRATCH.values()))
        build=next(v for k,v in fa._CACHE.items() if isinstance(k,tuple) and len(k)==6)
        obj,kern,blockdim=build
        handle=fa._CACHE[('hdl',obj)]
        stream=ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        acl_times=[]
        torch.npu.synchronize()
        for _ in range(args.reps):
            first=ctypes.c_void_p();last=ctypes.c_void_p()
            check(acl.aclrtCreateEventWithFlag(ctypes.byref(first),8),'create start event')
            check(acl.aclrtCreateEventWithFlag(ctypes.byref(last),8),'create end event')
            try:
                check(acl.aclrtRecordEvent(first,stream),'record start')
                check(acl.fa910a_launch(handle,kern.encode(),blockdim,
                    scratch['q'].data_ptr(),scratch['k'].data_ptr(),scratch['v'].data_ptr(),
                    scratch['o'].data_ptr(),scratch['l'].data_ptr(),stream),'ACL kernel launch')
                check(acl.aclrtRecordEvent(last,stream),'record end')
                check(acl.aclrtSynchronizeEventWithTimeout(last,60000),'wait end event')
                ms=ctypes.c_float()
                check(acl.aclrtEventElapsedTime(ctypes.byref(ms),first,last),'elapsed time')
                acl_times.append(float(ms.value))
            finally:
                check(acl.aclrtDestroyEvent(first),'destroy start event')
                check(acl.aclrtDestroyEvent(last),'destroy end event')
        report['kernel_event_ms']=acl_times
        report['kernel_event_median_ms']=statistics.median(acl_times)
        report['kernel_timing_method']='native ACL timing events on exact ctypes launch stream, NZ inputs prebuilt'
        report['direct_acl_timed_launches']=args.reps
        fa._LIB=acl
        repeat_hashes=[]
        for _ in range(3):
            repeat_hashes.append(digest(fa.flash_attention_910a(*inputs,n_pad=args.pad)))
        report['output_repeat_sha256']=repeat_hashes
        report['output_bit_repeatable']=all(h==report['output_sha256'] for h in repeat_hashes)
        report['input_sha256_after']=[digest(t) for t in inputs]
        assert report['input_sha256_after']==report['input_sha256'], 'input mutation'
        report['peak_allocated_bytes']=int(torch.npu.max_memory_allocated())
        report['peak_reserved_bytes']=int(torch.npu.max_memory_reserved())
        kernel_dir=Path(os.environ['H3_FA215_KERNEL_DIR'])
        entries=[]
        for file in sorted(kernel_dir.rglob('*.o')):
            meta=json.loads(file.with_suffix('.json').read_text())
            entries.append({'object':str(file),'object_bytes':file.stat().st_size,
                            'object_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                            'symbol':file.stem+'__kernel0','blockDim':meta.get('blockDim'),
                            'instance':meta.get('r217_actual_instance')})
        report['compiled_kernels']=entries
        assert len(entries)==1, entries
        assert entries[0]['instance'] is not None, 'no actual build record'
        instance=entries[0]['instance']
        assert instance['key_seq_limit']==report['valid_keys']
        assert instance['core_num']==30
        assert not instance['has_attn_mask'], 'unexpected attention mask'
        report['complete']=True
        # Output stays on host, for real direct comparison (no upload to client).
        torch.save(value.detach().cpu(), output/'output_host_only.pt')
    except BaseException as exc:
        report['error']=repr(exc)
        report['traceback']=traceback.format_exc()
        print(report['traceback'],flush=True)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('R218_RESULT '+json.dumps({k:v for k,v in report.items() if k not in ('traceback','compiled_kernels')}),flush=True)
    return 0 if report['complete'] else 2

if __name__=='__main__':
    raise SystemExit(main())
