#!/usr/bin/env python3
"""Scoped single-launch Float-accumulating vision attention; source awaiting real validation."""
from pathlib import Path
import hashlib
import copy
import math

CANDIDATE_SHA='2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78'
RUNTIME_SHA='c8890dbcd825fe9177c7ee8d5c6b46e4d00231d4d61e6421f83491fec85c52a6'
LIBRARY_SHA='bd50d7005bf48e9a9a853c01750eaea4eefbb3e0f69abb2e7153aa7139e8b043'
MAX_SCORE_BOUND=16384.0
HELPER='_vision_sdpa_bf16_compatible'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FlashVisionAttention105:
    def __init__(self,candidate,runtime,kernel_dir):
        self._candidate=candidate;self._runtime=runtime;self._dir=Path(kernel_dir)
        self._original=getattr(candidate,HELPER);assert callable(self._original)
        self._meta=None;self._enabled=False;self._installed=True
        self._stats={'calls_disabled':0,'calls_enabled':0,'kernel_launches':0,'Cauchy_guard_used':0}
        self._last_bound=None;setattr(candidate,HELPER,self.forward)

    def prepare(self):
        import sys
        import flash_vision105
        assert sha(self._candidate.__file__)==CANDIDATE_SHA
        assert sha(sys.modules[type(self._runtime).__module__].__file__)==RUNTIME_SHA
        assert self._runtime.snapshot()['library_sha256']==LIBRARY_SHA
        self._meta=flash_vision105.build_kernel(batch=16,m=4032,n=4032,d=72,out_dir=self._dir)
        assert sha(self._meta['obj_path'])==self._meta['obj_sha256']
        assert self._meta['block_dim']==30
        self._enabled=False
        # The returned witness cannot be emptied by later uninstall().
        return {'kernel':copy.deepcopy(self._meta),'source_SHA256':sha(flash_vision105.__file__),
                'engine_SHA256':sha(__file__),'original_candidate_SHA256':CANDIDATE_SHA,
                'gm_order':'QHalfNZ,KHalfNZ,VHalfNZ,FloatAuxUnused,HalfOutNZ',
                'score_abs_bound_required':MAX_SCORE_BOUND,'no_pixel_or_hidden_conditioning_cache':True}

    def set_enabled(self,value):
        assert isinstance(value,bool)
        if value and self._meta is None:raise RuntimeError('prepare required before enabling flash vision')
        self._enabled=value;return value

    def forward(self,q,k,v):
        if not self._enabled:
            self._stats['calls_disabled']+=1;return self._original(q,k,v)
        import torch
        assert q.shape==k.shape==v.shape==(1,16,4032,72)
        assert q.dtype==k.dtype==v.dtype==torch.float16 and q.device==k.device==v.device
        assert q.device.type in('npu','privateuseone')and q.device.index==self._runtime.device
        # Frozen tensors can be views with nonzero storage offset. Make the
        # numeric-guard operands dense, offset-zero; preserve every Half value.
        clean=[]
        for value in (q,k,v):
            dense=value.contiguous()
            if dense.storage_offset()!=0:
                dense=dense.clone(memory_format=torch.contiguous_format)
            clean.append(dense)
        q,k,v=clean
        # A Half max is ONLY a softmax shift estimate. Bound it tightly enough
        # that unnormalised exp high/low neither overflows nor loses its peak.
        qmax=float(q.abs().reshape(-1).amax(dim=0).item());kmax=float(k.abs().reshape(-1).amax(dim=0).item())
        bound=math.sqrt(72)*qmax*kmax
        if math.isfinite(bound)and bound>=MAX_SCORE_BOUND:
            qnorm=float(q.float().square().sum(-1).sqrt().reshape(-1).amax(dim=0).item())
            knorm=float(k.float().square().sum(-1).sqrt().reshape(-1).amax(dim=0).item())
            bound=qnorm*knorm*(72**-0.5)*(1+1e-4)
            self._stats['Cauchy_guard_used']+=1
        if not math.isfinite(bound)or bound>=MAX_SCORE_BOUND or not bool(torch.isfinite(v).reshape(-1).all(dim=0).item()):
            raise RuntimeError('input outside the validated finite Half-shift/dual-Half-exp domain; no clamp')
        self._last_bound=bound;device=q.device
        packed=[]
        for original in(q,k):
            rows=4032 if len(packed)==0 else 4096
            padded=torch.zeros((16,rows,80),dtype=torch.float16,device=device);padded[:,:4032,:72]=original[0]
            # Only K/V pad to4096; query rows remain exactly4032.
            packed.append(padded.reshape(16,rows,5,16).permute(0,2,1,3).contiguous())
        padded_v=torch.zeros((16,4096,80),dtype=torch.float16,device=device);padded_v[:,:4032,:72]=v[0]
        packed_v=padded_v.reshape(16,256,16,80).permute(0,1,3,2).contiguous()
        aux=torch.empty((16,),dtype=torch.float32,device=device)
        output=torch.empty((16,5,4032,16),dtype=torch.float16,device=device)
        self._runtime.launch(self._meta,[packed[0],packed[1],packed_v,aux,output],synchronize=True)
        self._stats['calls_enabled']+=1;self._stats['kernel_launches']+=1
        return output.permute(0,2,1,3).reshape(16,4032,80)[...,:72].contiguous().unsqueeze(0)

    def snapshot(self):
        return {'round':105,'enabled':self._enabled,'installed':self._installed,
                'kernel':copy.deepcopy(self._meta),'stats':dict(self._stats),
                'last_absolute_scaled_QK_bound':self._last_bound,'maximum_bound':MAX_SCORE_BOUND,
                'query_rows':4032,'key_rows_original':4032,'key_rows_padded':4096,'KV_tile':128,'last64keys_excluded_from_softmax_denominator':True,'head_dim_original':72,'head_dim_pad':80,
                'Half_max_only_estimate_Float_scores_unchanged':True,
                'one_kernel_launch_per_image':True,'general_Float_rescale_and_denominator_broadcast_exact_bits_no_Half_narrowing':True,'Float_cumulative_output_and_denominator':True,
                'input_QKV_not_cached_across_calls':True,'no_FloatLinear_weights_or_BFgoldcast':True,
                'full_encoder_precision_accepted':False,'whole_goal_complete':False,
                'numerical_tolerance':None,'source_SHA256':sha(__file__)}

    def uninstall(self):
        if self._installed:
            current=getattr(self._candidate,HELPER)
            if getattr(current,'__self__',None)is not self or getattr(current,'__func__',None)is not type(self).forward:
                raise RuntimeError('refusing to clobber another helper patch')
            setattr(self._candidate,HELPER,self._original)
        self._enabled=False;self._installed=False
        result=self.snapshot();self._meta=None;result['uninstalled']=True;return result
