"""R218 independent trim FA measurement with a fail-closed real-input leaf gate.

The first forward returns only the original FA output, while comparing every original
and trim output word on identical actual Q/K/V. No relaxed numerical tolerance, no
other candidate and no permanent runtime/source edits are introduced.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import torch

class TrimStage218:
    def __init__(self):
        from vllm_omni.diffusion.attention.backends import fa910a
        import fa218trim
        self.original_module=fa910a
        self.candidate=fa218trim
        self.original=fa910a.flash_attention_910a
        assert hashlib.sha256(Path(inspect.getfile(fa910a)).read_bytes()).hexdigest()=='937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'
        self.phase='inactive';self.rows=[];self.calls={'gate':0,'measurement':0}
        self.launches={'gate':0,'measurement':0};self.failed=False;self.installed=False
        self.lib=fa218trim._lib()
        self.launch_original=self.lib.fa910a_launch
        def launch(*args):
            assert self.phase in ('gate','measurement'),self.phase
            rc=self.launch_original(*args)
            if rc==0:self.launches[self.phase]+=1
            return rc
        self.launch_wrapper=launch
        self.lib.fa910a_launch=launch
        self.bound=self._attention

    def _attention(self,*args,**kwargs):
        q=args[0]
        assert list(q.shape)==[1,7,41792,128] and q.dtype==torch.float16
        assert kwargs.get('n_pad',0)==23
        assert self.phase in ('gate','measurement')
        self.calls[self.phase]+=1
        if self.phase=='measurement':
            return self.candidate.flash_attention_910a(*args,**kwargs)
        reference=self.original(*args,**kwargs)
        candidate=self.candidate.flash_attention_910a(*args,**kwargs)
        # Integer comparison is independent of defective Half isnan/isfinite kernels.
        a=reference.contiguous().view(torch.int16)
        b=candidate.contiguous().view(torch.int16)
        same=bool(torch.equal(a,b))
        different=0 if same else int((a!=b).sum(dtype=torch.int32).cpu())
        row={'call':self.calls['gate'],'elements':reference.numel(),
             'differing_words':different,'bit_identical':same,'shape':list(q.shape),'n_pad':23}
        if not same:
            af=reference.float();bf=candidate.float()
            row['relative_rms']=float(torch.linalg.vector_norm((af-bf).flatten()).cpu())/max(float(torch.linalg.vector_norm(af.flatten()).cpu()),1e-30)
            self.failed=True
        self.rows.append(row)
        # The gate forward ALWAYS uses original results, even if a mismatch is found.
        return reference

    def begin_gate(self):
        assert not self.installed
        self.phase='gate';self.original_module.flash_attention_910a=self.bound;self.installed=True

    def end_gate(self):
        assert self.calls['gate']==self.launches['gate']==50
        return {'actual_FA_calls':50,'actual_candidate_ACL_launches':self.launches['gate'],
                'compared_elements':sum(r['elements'] for r in self.rows),
                'differing_words':sum(r['differing_words'] for r in self.rows),
                'passed':not self.failed,'per_call':self.rows,
                'gate_forward_returned_original_outputs':True}

    def enable_measurement(self):
        assert self.installed and not self.failed
        self.phase='measurement'

    def restore(self):
        if self.installed:
            assert self.original_module.flash_attention_910a is self.bound
            self.original_module.flash_attention_910a=self.original
            self.installed=False
        self.phase='inactive'

    def report(self):
        kernels=[]
        for key,value in self.candidate._CACHE.items():
            if isinstance(key,tuple) and len(key)==6 and isinstance(key[0],int):
                obj,symbol,blocks=value
                meta=json.loads(Path(obj).with_suffix('.json').read_text())
                kernels.append({'path':obj,'symbol':symbol,'blockDim':blocks,
                                'sha256':hashlib.sha256(Path(obj).read_bytes()).hexdigest(),
                                'instance':meta.get('r217_actual_instance')})
        return {'calls':self.calls,'actual_ACL_launches':self.launches,'gate_failed':self.failed,
                'candidate_kernel':kernels,'nothing_adopted':True,
                'only_trim_FA_changed':True,'all_strict_checks_and_FP32_guards_preserved':True,
                'restored':not self.installed}

    def close(self):
        self.restore()
        assert self.lib.fa910a_launch is self.launch_wrapper
        self.lib.fa910a_launch=self.launch_original
