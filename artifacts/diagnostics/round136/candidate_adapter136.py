"""Transactionally route only the unique final-AdaLN layer to the approved candidate.

A method instance may be shared by other layers: every unknown actual layer is
strictly delegated to its old bound apply. Production adoption is not approved;
this adapter is controlled only by the private full49 quality-comparison RPC.
"""
from fp32_bias_candidate136 import ExplicitFP32Bias136


class FinalAdaLNCandidate136:
    def __init__(self,torch,model):
        self.torch=torch;self.target=model.final_layer.adaln_proj.linear
        self.method=self.target.quant_method;self.old=self.method.apply
        self.had_override='apply' in self.method.__dict__;self.override=self.method.__dict__.get('apply')
        self.candidate=ExplicitFP32Bias136(torch,self.target);self.installed=False;self.wrapper=None
        self.delegated_other_layers=0

    def install(self):
        assert not self.installed
        def route(layer,x,bias=None,*args,**kwargs):
            if layer is not self.target:
                self.delegated_other_layers+=1
                return self.old(layer,x,bias,*args,**kwargs)
            assert not args and not kwargs,'Require the original known final-AdaLN apply contract'
            return self.candidate.apply(layer,x,bias)
        self.wrapper=route
        try:
            self.method.apply=route;self.installed=True
            assert self.method.apply is route
        except BaseException:
            if self.had_override:self.method.apply=self.override
            elif 'apply' in self.method.__dict__:delattr(self.method,'apply')
            self.installed=False;raise

    def restore(self):
        if not self.installed:return
        assert self.method.apply is self.wrapper,'Do not overwrite an unexpected method mutation'
        if self.had_override:self.method.apply=self.override
        else:delattr(self.method,'apply')
        self.installed=False;self.wrapper=None
        assert ('apply' in self.method.__dict__)==self.had_override

    def physically_restored(self):
        if self.installed:return False
        return ('apply' in self.method.__dict__)==self.had_override and (not self.had_override or self.method.__dict__['apply'] is self.override)
