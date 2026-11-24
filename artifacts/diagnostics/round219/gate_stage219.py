"""R219 gate-only comparison of R218 trim kernel; never returns candidate results.

Every compared layer uses identical real Q/K/V. No candidate arithmetic is fed into
model state, no numeric tolerance is used, and a mismatch stops further candidate calls.
"""
from __future__ import annotations
import hashlib
import inspect
import json
from pathlib import Path
import torch

class GateStage219:
    def __init__(self):
        from vllm_omni.diffusion.attention.backends import fa910a
        import fa218trim
        self.native_module=fa910a
        self.candidate=fa218trim
        self.original=fa910a.flash_attention_910a
        self.native_sha=hashlib.sha256(Path(inspect.getfile(fa910a)).read_bytes()).hexdigest()
        assert self.native_sha=='937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'
        self.candidate_sha=hashlib.sha256(Path(inspect.getfile(fa218trim)).read_bytes()).hexdigest()
        assert self.candidate_sha=='52365cbf64180e9766b4dfc4c3a7bff5f311b6b8f8af7fc08b340a2bf8487b73'
        self.bound=self._attention
        self.installed=False;self.closed=False;self.failed=False
        self.rows=[];self.step=None;self.original_calls=0;self.candidate_calls=0
        self.launch_count=0;self.all_reports=[]
        self.lib=fa218trim._lib()
        self.native_launch=self.lib.fa910a_launch
        def launch(*args):
            assert self.installed
            rc=self.native_launch(*args)
            if rc==0:self.launch_count+=1
            return rc
        self.launch=launch
        self.lib.fa910a_launch=self.launch

    @staticmethod
    def _nonfinite_fp16_words(value):
        # Integer IEEE check, independent of Half isnan/isfinite implementation.
        words=value.contiguous().view(torch.int16).to(torch.int32)
        magnitude=torch.bitwise_and(words,0x7fff)
        return int((magnitude>=0x7c00).sum(dtype=torch.int32).cpu())

    def _attention(self,*args,**kwargs):
        q=args[0]
        assert list(q.shape)==[1,7,41792,128] and q.dtype==torch.float16
        assert kwargs.get('n_pad',0)==23
        self.original_calls+=1
        reference=self.original(*args,**kwargs)
        if self.failed:
            # Fail closed: no further candidate execution, original path continues.
            return reference
        candidate=self.candidate.flash_attention_910a(*args,**kwargs)
        self.candidate_calls+=1
        a=reference.contiguous().view(torch.int16)
        b=candidate.contiguous().view(torch.int16)
        same=bool(torch.equal(a,b))
        different=0 if same else int((a!=b).sum(dtype=torch.int32).cpu())
        row={'step':self.step,'layer_index':self.original_calls-1,
             'elements':reference.numel(),'shape':list(q.shape),'n_pad':23,
             'bit_identical':same,'differing_words':different}
        if not same:
            af=reference.float();bf=candidate.float()
            row['relative_rms']=float(torch.linalg.vector_norm((af-bf).flatten()).cpu())/max(float(torch.linalg.vector_norm(af.flatten()).cpu()),1e-30)
            row['reference_nonfinite_words']=self._nonfinite_fp16_words(reference)
            row['candidate_nonfinite_words']=self._nonfinite_fp16_words(candidate)
            self.failed=True
        self.rows.append(row)
        # Gate output cannot replace model math, even if comparison passed.
        return reference

    def begin_gate(self,step):
        assert not self.installed and not self.closed
        self.step=step;self.rows=[];self.original_calls=0;self.candidate_calls=0;self.launch_count=0
        self.native_module.flash_attention_910a=self.bound;self.installed=True

    def end_gate(self):
        assert self.original_calls==50
        assert self.candidate_calls==self.launch_count
        if not self.failed:assert self.candidate_calls==50
        report={'step':self.step,'original_FA_calls':self.original_calls,
                'candidate_FA_calls':self.candidate_calls,'candidate_ACL_launches':self.launch_count,
                'compared_elements':sum(row['elements'] for row in self.rows),
                'differing_words':sum(row['differing_words'] for row in self.rows),
                'passed':not self.failed,'candidate_comparison_stopped_on_failure':True,
                'gate_forward_returned_original_outputs':True,
                'no_candidate_result_used_in_model':True,'per_layer':list(self.rows)}
        self.all_reports.append(report)
        return report

    def restore(self):
        if self.installed:
            assert self.native_module.flash_attention_910a is self.bound
            self.native_module.flash_attention_910a=self.original
            self.installed=False

    def report(self):
        kernels=[]
        for key,value in self.candidate._CACHE.items():
            if isinstance(key,tuple) and len(key)==6 and isinstance(key[0],int):
                obj,symbol,blocks=value
                meta=json.loads(Path(obj).with_suffix('.json').read_text())
                kernels.append({'symbol':symbol,'blockDim':blocks,
                                'sha256':hashlib.sha256(Path(obj).read_bytes()).hexdigest(),
                                'instance':meta.get('r217_actual_instance')})
        return {'gate_failed':self.failed,'restored':not self.installed,
                'source_sha256':{'native':self.native_sha,'candidate':self.candidate_sha},
                'candidate_kernel':kernels,'gate_reports':self.all_reports,
                'candidate_results_returned_to_model':0,'candidate_measurements_or_adoption':False,
                'no_combo_with_other_candidate':True}

    def close(self):
        if self.closed:return
        self.restore()
        assert self.lib.fa910a_launch is self.launch
        self.lib.fa910a_launch=self.native_launch
        self.closed=True
