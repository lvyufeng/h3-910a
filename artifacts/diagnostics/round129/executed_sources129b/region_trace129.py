"""Bounded output fingerprints and first-changing-region trace, always original returns.

No activation replacement or model math changes. SHA fingerprints locate a
region but are not complete per-word equality or a pure device-time profile.
Snapshots for detailed paired proof are limited to one selected region/pass.
"""
from __future__ import annotations
from types import MethodType
from collections import defaultdict
from fingerprints129 import fingerprint,normalized_fingerprint,digest


class RegionTrace129:
    def __init__(self,torch,model,*,detailed_block=None):
        self.torch=torch;self.model=model;self.rows=[];self.ordinal=defaultdict(int)
        self.handles=[];self.methods=[];self.detailed_block=detailed_block
        paths={'video_patch_proj','audio_patch_proj','condition_proj','time_embedder','rope','token_refiner','sp_prepare','sp_gather','final_layer'}
        for index in range(50):paths.add(f'blocks.{index}')
        if detailed_block is not None:
            prefix=f'blocks.{detailed_block}'
            for name,module in model.named_modules():
                if name.startswith(prefix+'.'):paths.add(name)
        for name,module in model.named_modules():
            if name not in paths:continue
            def pre(owner,args,kwargs,name=name):
                count=self.ordinal[name];self.ordinal[name]+=1
                self.rows.append({'module':name,'ordinal':count,'boundary':'input',
                    'leaves':normalized_fingerprint(fingerprint({'args':args,'kwargs':kwargs},torch)),
                    'D2H_trace_synchronizes_current_work_not_clean_timing':True})
            def post(owner,args,kwargs,result,name=name):
                count=self.ordinal[name]-1
                self.rows.append({'module':name,'ordinal':count,'boundary':'output',
                    'leaves':normalized_fingerprint(fingerprint(result,torch)),
                    'D2H_trace_synchronizes_current_work_not_clean_timing':True})
                return None  # nn.Module hook preserves the exact original result object.
            self.handles.append(module.register_forward_pre_hook(pre,with_kwargs=True))
            self.handles.append(module.register_forward_hook(post,with_kwargs=True))
        old=model._embed
        def embed(instance,*args,**kwargs):
            count=self.ordinal['_embed'];self.ordinal['_embed']+=1
            self.rows.append({'module':'_embed','ordinal':count,'boundary':'input','leaves':normalized_fingerprint(fingerprint({'args':args,'kwargs':kwargs},torch))})
            result=old(*args,**kwargs)
            self.rows.append({'module':'_embed','ordinal':count,'boundary':'output','leaves':normalized_fingerprint(fingerprint(result,torch))})
            return result
        wrapped=MethodType(embed,model);model._embed=wrapped;self.methods.append((model,'_embed',old,wrapped))

    def reset(self):self.rows=[];self.ordinal.clear()

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles.clear()
        for owner,name,old,wrapper in self.methods:
            assert getattr(owner,name) is wrapper;setattr(owner,name,old)
        self.methods.clear()

    @staticmethod
    def compare(reference,candidate):
        assert [(r['module'],r['ordinal'],r['boundary']) for r in reference]==[(r['module'],r['ordinal'],r['boundary']) for r in candidate]
        compared=[]
        for a,b in zip(reference,candidate):
            same=digest(a['leaves'])==digest(b['leaves'])
            compared.append({'module':a['module'],'ordinal':a['ordinal'],'boundary':a['boundary'],'all_leaf_SHA_equal':same})
        first=next((r for r in compared if not r['all_leaf_SHA_equal']),None)
        return {'first_different_trace_boundary':first,'comparisons':compared,
                'hash_only_not_different_word_count':True,'trace_can_change_race_timing':True}
