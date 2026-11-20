#!/usr/bin/env python3
"""Test native and same-semantics Half NanToNum on the owned original910A.

Tiny operator unit tests only, not CPU model inference or a GPU/NPU generation
benchmark. Both probes run in separate processes so an unavailable native CANN
kernel cannot poison the compatibility probe's NPU context. Half signed zeros,
subnormals, infinity bounds, explicit replacements, out and inplace are checked.
"""
import argparse
import json
from pathlib import Path
import time

ROOT=Path('/tmp/h3_align12401')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['native','equivalent'],required=True)
    parser.add_argument('--tag',default='124g')
    args=parser.parse_args()
    import torch
    import torch_npu
    torch.npu.set_device(0);torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    from npu_half_nan_compat124 import half_nan_scope
    result={'scope':__doc__,'mode':args.mode,'completed':False,'CPU_model_inference':False,'BF16_used':False,'checks':[]}
    path=ROOT/f'Half_nan_to_num_{args.mode}_probe{args.tag}.json';assert not path.exists()
    try:
        with half_nan_scope(enabled=args.mode=='equivalent') as counter:
            for dtype in [torch.float16,torch.float32]:
                bound=torch.finfo(dtype).max;small=2**-24 if dtype==torch.float16 else 2**-149
                cpu=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.,-1.,bound,-bound,small,-small,torch.finfo(dtype).tiny],dtype=dtype,device='cpu')
                npu=cpu.to('npu:0')
                cases=[{}, {'nan':-2.,'posinf':3.,'neginf':-4.}, {'nan':1.00048828225,'posinf':0.,'neginf':-0.}]
                for values in cases:
                    expected=torch.nan_to_num(cpu,**values)
                    outputs=[('function',torch.nan_to_num(npu,**values)),('method',npu.nan_to_num(**values)),
                             ('inplace',npu.clone().nan_to_num_(**values))]
                    out=torch.empty_like(npu);torch.nan_to_num(npu,out=out,**values);outputs.append(('out',out))
                    for name,tensor in outputs:
                        actual=tensor.cpu();bits=torch.int16 if dtype==torch.float16 else torch.int32
                        equal=torch.equal(actual.view(bits),expected.view(bits))
                        result['checks'].append({'api':name,'replacements':values,'bit_identical_to_native_CPU_operator':equal,'output_dtype':str(actual.dtype)})
                        assert equal,(name,values,actual,expected)
            torch.npu.synchronize()
        result['compatibility_call_count']=counter['calls'];result['completed']=True
    except BaseException as exc:
        result['failure']={'type':type(exc).__name__,'message':str(exc)}
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if args.mode=='equivalent' and not result['completed']:raise SystemExit(1)


if __name__=='__main__':main()
