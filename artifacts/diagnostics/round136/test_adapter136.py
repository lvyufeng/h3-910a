#!/usr/bin/env python3
"""Pure mock: same FP32-output arithmetic on original 2/3 rows and target routing."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from fp32_bias_candidate136 import ExplicitFP32Bias136
import candidate_adapter136 as adapter_module

ROOT=Path('/tmp/h3_quality13601');output=ROOT/'adapter_unit136.json';assert not output.exists()
log=[];device=SimpleNamespace(type='npu')
class Tensor:
    def __init__(self,shape,dtype):self.shape=shape;self.ndim=len(shape);self.dtype=dtype;self.device=device
    def t(self):log.append('weight_transpose_view');return Tensor(tuple(reversed(self.shape)),self.dtype)
    def float(self):log.append('bias_FP32_expansion');return Tensor(self.shape,'float32')
    def add_(self,value):assert self.dtype==value.dtype=='float32';log.append('FP32_add');return self
    def to(self,dtype):assert dtype=='float16';log.append('one_Half_narrowing');return Tensor(self.shape,dtype)
class Torch:
    float16='float16';float32='float32'
    def mm(self,x,w,*,out_dtype):
        assert x.dtype==w.dtype=='float16' and w.shape==(2688,5376) and out_dtype=='float32'
        log.append('Half_GEMM_FP32_output');return Tensor((x.shape[0],5376),'float32')
layer=SimpleNamespace(weight=Tensor((5376,2688),'float16'));bias=Tensor((5376,),'float16');original_weight=layer.weight
candidate=ExplicitFP32Bias136(Torch(),layer)
for rows in (2,3):assert candidate.apply(layer,Tensor((rows,2688),'float16'),bias).shape==(rows,5376)
assert candidate.calls==2 and candidate.row_count_calls=={'2':1,'3':1} and layer.weight is original_weight
assert log==['weight_transpose_view','Half_GEMM_FP32_output','bias_FP32_expansion','FP32_add','one_Half_narrowing']*2
class Method:
    def apply(self,layer,x,bias=None,*args,**kwargs):return ('original',layer,x,bias,args,kwargs)
class Candidate:
    def __init__(self,torch,target):self.target=target;self.calls=0
    def apply(self,layer,x,bias=None):assert layer is self.target;self.calls+=1;return ('candidate',layer,x,bias)
method=Method();target=SimpleNamespace(quant_method=method);other=SimpleNamespace(quant_method=method)
model=SimpleNamespace(final_layer=SimpleNamespace(adaln_proj=SimpleNamespace(linear=target)))
with patch.object(adapter_module,'ExplicitFP32Bias136',Candidate):
    adapter=adapter_module.FinalAdaLNCandidate136(object(),model)
    try:
        adapter.install();assert method.apply(target,'x','b')[0]=='candidate'
        assert method.apply(other,'x','b','arg',key='old')==('original',other,'x','b',('arg',),{'key':'old'})
        raise RuntimeError('Injected failure')
    except RuntimeError:pass
    finally:adapter.restore()
assert adapter.physically_restored() and 'apply' not in method.__dict__
result={'completed':True,'pure_mock_no_Torch_NPU_model_arithmetic':True,'original_2_and3_row_shapes_preserved':True,
 'same_Half_operands_FP32_output_bias_one_Half_narrowing_call_order':True,'unique_target_only_routed':True,
 'unknown_shared_method_args_and_original_return_preserved':True,'exception_descriptor_restored':True,
 'not_numeric_equivalence_repeatability_or_runtime_support_proof':True}
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
