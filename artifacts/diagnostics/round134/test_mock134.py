#!/usr/bin/env python3
"""Pure mock unit: authorized candidate call order/dtypes, not numerical inference."""
from pathlib import Path
import json
from types import SimpleNamespace
from fp32_bias_candidate134 import ExplicitFP32Bias134

ROOT=Path('/tmp/h3_adaln13401');path=ROOT/'mock_unit134.json';assert not path.exists()
log=[];device=SimpleNamespace(type='npu')
class Tensor:
    def __init__(self,shape,dtype):self.shape=shape;self.dtype=dtype;self.device=device
    def t(self):log.append('weight_transpose_view');return Tensor((self.shape[1],self.shape[0]),self.dtype)
    def float(self):log.append('bias_exact_FP32_expansion');return Tensor(self.shape,'torch.float32')
    def add_(self,value):assert self.dtype==value.dtype=='torch.float32';log.append('FP32_add');return self
    def to(self,dtype):assert dtype=='torch.float16';log.append('one_Half_writeback');return Tensor(self.shape,dtype)
class Torch:
    float16='torch.float16';float32='torch.float32'
    def mm(self,x,w,*,out_dtype):
        assert x.dtype==w.dtype==self.float16 and out_dtype==self.float32
        assert x.shape==(3,2688) and w.shape==(2688,5376)
        log.append('Half_GEMM_FP32_output');return Tensor((3,5376),self.float32)
layer=SimpleNamespace(weight=Tensor((5376,2688),'torch.float16'));bias=Tensor((5376,),'torch.float16');x=Tensor((3,2688),'torch.float16')
original_weight=layer.weight;candidate=ExplicitFP32Bias134(Torch(),layer);result=candidate.apply(layer,x,bias)
assert log==['weight_transpose_view','Half_GEMM_FP32_output','bias_exact_FP32_expansion','FP32_add','one_Half_writeback']
assert result.dtype=='torch.float16' and layer.weight is original_weight and candidate.calls==1
try:candidate.apply(SimpleNamespace(weight=layer.weight),x,bias)
except AssertionError:pass
else:raise AssertionError('Unknown layer should be rejected before arithmetic')
assert candidate.calls==1
row={'completed':True,'Torch_NPU_or_model_inference':False,'pure_mock_call_order_dtype_contract_verified':True,
    'same_parameter_identity_preserved':True,'wrong_layer_refused':True,'numeric_equivalence_or_repeatability_proven':False}
path.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
