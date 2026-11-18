#!/usr/bin/env python3
"""Private explicitFloat erf formula for the four original merger GELUs only."""
from pathlib import Path
import hashlib
from types import MethodType
NAMES=('merger.act_fn','deepstack_merger_list.0.act_fn','deepstack_merger_list.1.act_fn','deepstack_merger_list.2.act_fn')


class TorchErfAdapter90:
    def __init__(self,vision):
        import torch
        all_modules=dict(vision.named_modules())
        self.entries={};self.enabled=False;self.calls=0;self.installed=True
        outer=self
        def make_forward(original):
            def forward(module,input_):
                if not outer.enabled:return original(input_)
                assert input_.dtype==torch.float16 and input_.device.type in('npu','privateuseone')
                assert input_.shape==(2016,4608)
                outer.calls+=1
                x=input_.float()
                return(.5*x*(1+torch.erf(x*(2**-.5)))).to(input_.dtype)
            return forward
        for name in NAMES:
            module=all_modules[name]
            assert type(module)is torch.nn.GELU and module.approximate=='none'
            assert 'forward'not in module.__dict__
            original=module.forward;function=make_forward(original)
            module.forward=MethodType(function,module)
            self.entries[name]=(module,original,function)
    def set_enabled(self,value):
        assert self.installed and isinstance(value,bool);self.enabled=value
    def snapshot(self):
        return{'enabled':self.enabled,'installed':self.installed,'calls':self.calls,
               'patched_modules':list(self.entries),'original_erf_modes_preserved':True,
               'no_MLP_tanh_or_other_functional_patch':True,'no_Float_weight_teacher':True,
               'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def uninstall(self):
        if self.installed:
            for module,original,function in self.entries.values():
                assert getattr(module.forward,'__self__',None)is module
                assert getattr(module.forward,'__func__',None)is function
                delattr(module,'forward')
            self.installed=False;self.enabled=False
