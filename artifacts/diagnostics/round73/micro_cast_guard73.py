#!/usr/bin/env python3
"""Actual910A full-word stridedcast/layout/readonly micro on a separate leased card."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from evidence_io73 import EvidenceWriter
from vision_cast_guard73 import cast

ROOT=Path('/tmp/h3_round73_cast02');writer=EvidenceWriter(ROOT/'micro01.json')
device=int(os.environ.get('H3_CAST_DEVICE','1'));torch.npu.set_device(device);torch.set_num_threads(4)
r={'scope':__doc__,'device_index':device,'cases':[],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def raw(v):
    x=v.detach().cpu().contiguous();return hashlib.sha256(memoryview(x.view(torch.uint8).numpy())).hexdigest()


def layout(v):return {'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype),'physical_format':torch_npu.get_npu_format(v)}


def timed(fn,v):
    torch.npu.synchronize();t=time.perf_counter();out=fn(v);torch.npu.synchronize();return out,time.perf_counter()-t


try:
    with torch.inference_mode():
        for n in(1,5,61,4032):
            for pitch in(1152,3456):
                offset=0 if pitch==1152 else 2304
                count=offset+n*pitch
                cpu=((torch.arange(count,dtype=torch.int32)*257+31)&65535).to(torch.uint16).view(torch.bfloat16)
                source=cpu.to(torch.device('npu',device))
                value=source.as_strided((1,16,n,72),(n*pitch,72,pitch,1),offset)
                before=raw(source)
                native=lambda x:x.float().contiguous()
                expected,_=timed(native,value);actual,first_s=timed(cast,value)
                a,b=actual.cpu().contiguous(),expected.cpu().contiguous()
                equal=torch.equal(a.view(torch.uint8),b.view(torch.uint8))
                row={'tokens':n,'input_pitch':pitch,'input_storage_offset':offset,'input_stride':list(value.stride()),
                     'elements':value.numel(),'full_bits_equal':equal,'candidate_layout':layout(actual),'native_layout':layout(expected),
                     'source_readonly':before==raw(source),'first_s_excluded_compile':first_s,'warm':[]}
                r['cases'].append(row);writer.write(r)
                assert equal and row['candidate_layout']==row['native_layout'] and row['source_readonly']
                for _ in range(3):
                    expected,native_s=timed(native,value);actual,candidate_s=timed(cast,value)
                    bits=torch.equal(actual.cpu().contiguous().view(torch.uint8),expected.cpu().contiguous().view(torch.uint8))
                    assert bits
                    row['warm'].append({'native_s':native_s,'candidate_s':candidate_s,'full_bits_equal':bits})
                row['native_median_s']=statistics.median(x['native_s']for x in row['warm'])
                row['candidate_median_s']=statistics.median(x['candidate_s']for x in row['warm'])
                writer.write(r)
                print(json.dumps({'n':n,'pitch':pitch,'all_raw_words_equal':equal,'native_s':row['native_median_s'],'cast_s':row['candidate_median_s']}),flush=True)
    r['completed']=True
except BaseException as exc:
    r['failure']={'type':type(exc).__name__,'message':str(exc)};raise
finally:writer.write(r)
