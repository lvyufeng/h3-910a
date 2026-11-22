#!/usr/bin/env python3
"""Pure-mock target routing and exact descriptor restoration; no model arithmetic."""
from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch
import candidate_adapter135 as module

ROOT=Path('/tmp/h3_full5013501');path=ROOT/'adapter_unit135.json';assert not path.exists()
class Method:
    def apply(self,layer,x,bias=None,*args,**kwargs):return ('original',layer,x,bias,args,kwargs)
class Candidate:
    def __init__(self,torch,target):self.target=target;self.calls=0
    def apply(self,layer,x,bias=None):assert layer is self.target;self.calls+=1;return ('candidate',layer,x,bias)
method=Method();target=SimpleNamespace(quant_method=method);other=SimpleNamespace(quant_method=method)
model=SimpleNamespace(final_layer=SimpleNamespace(adaln_proj=SimpleNamespace(linear=target)))
with patch.object(module,'ExplicitFP32Bias134',Candidate):
    adapter=module.FinalAdaLNCandidate135(object(),model);assert adapter.physically_restored()
    try:
        adapter.install()
        assert method.apply(target,'x','b')[0]=='candidate'
        result=method.apply(other,'x','b','extra',tag='original_kwargs')
        assert result==('original',other,'x','b',('extra',),{'tag':'original_kwargs'})
        assert adapter.candidate.calls==1 and adapter.delegated_other_layers==1
        raise RuntimeError('Injected diagnostic error')
    except RuntimeError:pass
    finally:adapter.restore()
assert 'apply' not in method.__dict__ and method.apply(target,'x','b')[0]=='original'
assert adapter.physically_restored()
row={'completed':True,'Torch_NPU_or_model_arithmetic':False,'unique_target_layer_only_routed':True,
 'shared_method_other_layer_args_and_original_returns_preserved':True,'exception_cleanup_restores_exact_descriptor':True,
 'numerical_stability_or_equivalence_not_proven_by_mock':True}
path.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
