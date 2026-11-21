"""Private R126 VAE autocast operand cache; ORIGINAL FP32 parameters stay intact.

Only ordinary nn.Linear modules under the 36 decoder transformer blocks are
eligible. Front x_embedder, proj_out, normalization, rotary, residuals and all
other operations are untouched. Cached Half operands are tried only inside the
same original NPU Half autocast context and must first pass per-input raw-word
proofs. This is not a Half-parameter rewrite and not a fused VAE kernel.
"""
from __future__ import annotations
from types import MethodType
import threading
import torch
import torch.nn.functional as F
import torch_npu


def signature(t):
    if t is None:return None
    return (str(t.device),str(t.dtype),tuple(t.shape),tuple(t.stride()),t.data_ptr(),
            t._version,int(torch_npu.get_npu_format(t)),bool(t.requires_grad),id(t),
            t.storage_offset(),int(t.untyped_storage()._cdata))


class AutocastOperandCache126:
    def __init__(self, decoder, budget_bytes=6*1024**3):
        self.decoder=decoder
        self.budget=budget_bytes
        self.rows=[]
        self.enabled=False
        self.proof=False
        self.proof_rows=[]
        self.calls=0
        self.fallbacks=0
        self.prepared=False
        self.failed=False
        self.stream=None
        self.original_parameter_signatures={}
        for i,block in enumerate(decoder.transformer_blocks):
            for name,module in block.named_modules():
                if type(module) is torch.nn.Linear:
                    assert 'forward' not in vars(module)
                    self.rows.append({'name':f'transformer_blocks.{i}.{name}','module':module,
                                      'original_forward':module.forward})
        assert len(decoder.transformer_blocks)==36 and self.rows

    def prepare(self):
        assert not self.prepared and not self.enabled
        total=0
        for row in self.rows:
            m=row['module']
            assert m.weight.device.type=='npu' and m.weight.dtype==torch.float32
            assert m.bias is None or (m.bias.dtype==torch.float32 and m.bias.device==m.weight.device)
            total+=m.weight.numel()*2+(m.bias.numel()*2 if m.bias is not None else 0)
        assert total<=self.budget,('Explicit Half-operand cache cap exceeded',total,self.budget)
        self.stream=torch.npu.current_stream(self.rows[0]['module'].weight.device)
        self.producer_thread=threading.get_ident()
        with torch.no_grad():
            for row in self.rows:
                m=row['module']
                row['signature']=(signature(m.weight),signature(m.bias))
                row['half_weight']=m.weight.detach().to(torch.float16)
                row['half_bias']=m.bias.detach().to(torch.float16) if m.bias is not None else None
                row['cache_signature']=(signature(row['half_weight']),signature(row['half_bias']))
                def forward(module,x,*,row=row):return self._apply(row,module,x)
                row['wrapper']=MethodType(forward,m)
                m.forward=row['wrapper']
        self.original_parameter_signatures={name:signature(p) for name,p in self.decoder.named_parameters()}
        self.cache_bytes=total
        self.prepared=True
        torch.npu.synchronize()

    def _autocast_half(self):
        return torch.is_autocast_enabled('npu') and torch.get_autocast_dtype('npu')==torch.float16

    def _apply(self,row,module,x):
        if not self.enabled or not self._autocast_half():
            self.fallbacks+=1
            return row['original_forward'](x)
        if self.failed:raise FloatingPointError('A cached-operand leaf already failed')
        assert x.device==module.weight.device and x.device.type=='npu'
        assert not torch.is_grad_enabled(), 'Operand reuse is inference-only; preserve original parameter flags'
        assert threading.get_ident()==self.producer_thread, 'Cache thread ownership changed'
        assert torch.npu.current_stream(x.device)==self.stream,'Cache producer/consumer stream changed'
        assert (signature(module.weight),signature(module.bias))==row['signature'],'Original FP32 parameters changed'
        assert (signature(row['half_weight']),signature(row['half_bias']))==row['cache_signature'],'Private Half operand changed'
        self.calls+=1
        if self.proof:
            reference=row['original_forward'](x)
            native_repeat=row['original_forward'](x)
            candidate=F.linear(x,row['half_weight'],row['half_bias'])
            assert candidate.shape==reference.shape==native_repeat.shape and candidate.dtype==reference.dtype==native_repeat.dtype
            # Preserve each exact IEEE word, including signed zeros. A second
            # native evaluation quantifies leaf self-noise; it is NOT a tolerance.
            a=reference.detach().cpu().contiguous();b=candidate.detach().cpu().contiguous()
            native_cpu=native_repeat.detach().cpu().contiguous()
            self_equal=bool(torch.equal(a.view(torch.uint8),native_cpu.view(torch.uint8)))
            raw_equal=bool(torch.equal(a.view(torch.uint8),b.view(torch.uint8)))
            diff=(a.double()-b.double())
            self.proof_rows.append({'module':row['name'],'input_shape':list(x.shape),
                'input_dtype':str(x.dtype),'output_dtype':str(a.dtype),'output_shape':list(a.shape),
                'elements':a.numel(),'raw_words_equal':raw_equal,'native_self_repeat_words_equal':self_equal,
                'finite_all':bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.isfinite(native_cpu).all()),
                'max_abs_error':float(diff.abs().max()),
                'relative_RMS':float(diff.square().sum().sqrt()/a.double().square().sum().sqrt().clamp_min(1e-30)),
                'original_weight_format':row['signature'][0][6],
                'cached_weight_format':row['cache_signature'][0][6]})
            if not raw_equal or not self_equal:self.failed=True
            assert (signature(module.weight),signature(module.bias))==row['signature'], 'Native or cached evaluation mutated parameters'
            return reference
        return F.linear(x,row['half_weight'],row['half_bias'])

    def set_enabled(self,value):
        assert type(value) is bool and self.prepared
        if value and self.failed:raise FloatingPointError('Refuse a failed operand cache')
        self.enabled=value

    def begin_proof(self):
        assert self.prepared and not self.failed
        self.proof_rows.clear();self.proof=True;self.enabled=True

    def end_proof(self):
        self.proof=False;self.enabled=False
        touched={row['module'] for row in self.proof_rows}
        expected={row['name'] for row in self.rows}
        return {'leaf_comparisons':len(self.proof_rows),'all_eligible_modules_executed':touched==expected,
                'all_same_input_leaf_words_equal':not self.failed and bool(self.proof_rows) and all(r['finite_all'] for r in self.proof_rows),
                'all_leaf_outputs_finite':bool(self.proof_rows) and all(r['finite_all'] for r in self.proof_rows),
                'proof_returns_original_outputs':True,'rows':self.proof_rows}

    def snapshot(self):
        return {'cached_Linear_modules':len(self.rows),'cache_bytes':getattr(self,'cache_bytes',0),
                'cached_operand_call_count':self.calls,'native_fallback_call_count':self.fallbacks,
                'original_all_parameters_still_FP32':all(p.dtype==torch.float32 for p in self.decoder.parameters()),
                'original_parameters_unchanged':all(signature(p)==self.original_parameter_signatures[n]
                                                      for n,p in self.decoder.named_parameters()) if self.prepared else None,
                'original_parameter_rewrite':False,'norm_rotary_residual_front_final_untouched':True,
                'fused_VAE_kernel':False,'accepted_lossless':False}

    def close(self):
        if not self.prepared:return
        torch.npu.synchronize()
        for row in self.rows:
            module=row['module']
            assert module.forward is row['wrapper']
            delattr(module,'forward')
            row.pop('half_weight',None);row.pop('half_bias',None)
        self.enabled=False;self.prepared=False
