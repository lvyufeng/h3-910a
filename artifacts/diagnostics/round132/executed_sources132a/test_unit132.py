#!/usr/bin/env python3
"""CPU-only small word-metric/constructor-rollback unit; no NPU/model execution."""
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch
import torch
import audio_tail_trace132 as trace_module
from paired_words132 import compare_fp32

ROOT=Path('/tmp/h3_audio13201');target=ROOT/'unit132.json';assert not target.exists()
a=torch.tensor([0.,1.,-2.,3.,4.,5.],dtype=torch.float32);b=a.clone()
assert compare_fp32(a,b,torch,chunk_words=2)['all_words_equal']
b.view(torch.int32)[1]+=1
result=compare_fp32(b,a,torch,chunk_words=2)
assert result['different_FP32_words']==1 and result['total_FP32_words']==6 and result['max_abs_difference']>0
assert result['relative_RMS_percent']==100*result['relative_RMS_ratio']

class Handle:
    def __init__(self):self.removed=False
    def remove(self):self.removed=True
class Module:
    def __init__(self):self.handles=[]
    def register_forward_pre_hook(self,*args,**kwargs):h=Handle();self.handles.append(h);return h
    def register_forward_hook(self,*args,**kwargs):h=Handle();self.handles.append(h);return h
class Method:
    def apply(self,layer,x,bias=None):return x
class Projection(Module):
    def __init__(self):super().__init__();self.quant_method=Method();self.weight=object();self.bias=object()
class Positions:
    def cpu(self):return self
    def numel(self):return 414
    def tolist(self):return list(range(414))
video=Projection();audio=Projection();final=Module();final.video_out=video;final.audio_out=audio
modules={'sp_gather':Module(),'time_embedder':Module(),'final_layer':final,'final_layer.norm':Module(),
 'final_layer.adaln_proj':Module(),'final_layer.video_out':video,'final_layer.audio_out':audio}
model=SimpleNamespace(final_layer=final,named_modules=lambda:list(modules.items()))
fake_torch=SimpleNamespace(npu=SimpleNamespace(mem_get_info=lambda:(64*1024**3,64*1024**3)))
try:
    with patch.object(Path,'read_text',return_value='MemAvailable: 268435456 kB\n'),patch.object(trace_module,'fingerprint',side_effect=RuntimeError('Injected dispatcher inspection error')):
        trace_module.AudioTailTrace132(fake_torch,model,Positions(),41792)
except RuntimeError as exc:assert str(exc)=='Injected dispatcher inspection error'
else:raise AssertionError('Expected injected constructor failure')
assert all(h.removed for module in modules.values() for h in module.handles)
assert 'apply' not in video.quant_method.__dict__ and 'apply' not in audio.quant_method.__dict__
receipt={'completed':True,'NPU_or_model_inference':False,'bounded_chunk_full_word_count_verified':True,
    'one_changed_FP32_word_detected':True,'RMS_percent_ratio_units_correct':True,
    'partial_constructor_failure_rolls_back_all_handles':True,'original_primary_exception_preserved':True,
    'class_dispatch_not_modified_by_failed_constructor':True}
target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
