#!/usr/bin/env python3
"""Probe integer-bit IEEE classification on original910A, no model inference.

The previous float-predicate substitutions failed the tiny NaN/Inf regression.
Test native bit views and integer operations instead of changing precision or
silently accepting them. Signed zero and the smallest NaN payloads are included.
"""
import json
from pathlib import Path

ROOT=Path('/tmp/h3_align12401')


def main():
    import torch
    import torch_npu
    torch.npu.set_device(0)
    report={'scope':__doc__,'completed':False,'CPU_model_inference':False,'checks':[]}
    path=ROOT/'NPU_IEEE_bit_predicate_probe124j.json';assert not path.exists()
    try:
        for dtype,bits in [(torch.float16,[0x7c00,0xfc00,0x7c01,0xfc01,0x7fff,0xffff,0,0x8000,1,0x8001,0x3c00]),
                           (torch.float32,[0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fffffff,0xffffffff,0,0x80000000,1,0x80000001,0x3f800000])]:
            width=16 if dtype==torch.float16 else 32
            signed=[v-(1<<width) if v&(1<<(width-1)) else v for v in bits]
            cpu=torch.tensor(signed,dtype=torch.int16 if width==16 else torch.int32).view(dtype)
            x=cpu.to('npu:0');dense=torch_npu.npu_format_cast(x,2)
            words=dense.reshape(-1).view(torch.int16)
            if width==16:
                high=words.to(torch.int32);low=None;limit=0x7c00
            else:
                words=words.reshape(-1,2);low=words[:,0].to(torch.int32)&0xffff;high=words[:,1].to(torch.int32);limit=0x7f80
            magnitude=high&0x7fff
            if width==16:
                nan_mask=magnitude>limit;inf_mask=magnitude==limit
            else:
                nan_mask=(magnitude>limit)|((magnitude==limit)&(low!=0))
                inf_mask=(magnitude==limit)&(low==0)
            pos=inf_mask&(high>=0);neg=inf_mask&(high<0)
            actual=[nan_mask.cpu(),pos.cpu(),neg.cpu()]
            expected=[torch.isnan(cpu),torch.isposinf(cpu),torch.isneginf(cpu)]
            checks=[torch.equal(a,b) for a,b in zip(actual,expected)]
            report['checks'].append({'dtype':str(dtype),'integer_bit_nan_posinf_neginf_correct':checks,
                                     'actual_masks':[a.tolist() for a in actual],'expected_masks':[a.tolist() for a in expected],
                                     'input_raw_bits_preserved':torch.equal(x.cpu().view(torch.int16 if width==16 else torch.int32),cpu.view(torch.int16 if width==16 else torch.int32))})
            assert all(checks),report['checks'][-1]
        torch.npu.synchronize();report['completed']=True
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)}
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if not report['completed']:raise SystemExit(1)


if __name__=='__main__':main()
