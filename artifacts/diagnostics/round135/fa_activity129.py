"""Count actual FA call/ACL-launch/error branches; preserve every original return.

No new synchronizations, kernels, cache clear or fallback policy changes. Errors
are recorded and re-raised so the original caller still chooses its own fallback.
"""
from __future__ import annotations


class FAActivity129:
    def __init__(self,torch):
        from vllm_omni.diffusion.attention.backends import fa910a
        self.torch=torch;self.fa=fa910a;self.saved=[];self.rows=[]
        try:
            self._function(fa910a,'flash_attention_910a','FA.wrapper')
            self._function(fa910a._lib(),'fa910a_launch','FA.ACL_launch')
        except BaseException as primary:
            try:self.close()
            except BaseException as rollback:raise primary from rollback
            raise

    def _function(self,owner,name,label):
        old=getattr(owner,name)
        def wrapper(*args,**kwargs):
            row={'kind':label,'current_stream':int(self.torch.npu.current_stream().npu_stream)}
            if label=='FA.wrapper':
                row.update({'q_shape':list(args[0].shape),'dtype':str(args[0].dtype),
                    'n_pad':int(kwargs.get('n_pad',0)),'passed_stream':repr(kwargs.get('stream'))})
            else:
                row.update({'ACL_stream':int(args[-1].value),'blockDim':int(args[2])})
                assert row['ACL_stream']==row['current_stream']
            try:
                result=old(*args,**kwargs);row['completed']=True
                if label=='FA.ACL_launch':row['return_code']=int(result)
                return result
            except BaseException as exc:
                row['completed']=False;row['exception']={'type':type(exc).__name__,'message':str(exc)};raise
            finally:self.rows.append(row)
        setattr(owner,name,wrapper);self.saved.append((owner,name,old,wrapper))

    def reset(self):self.rows=[]

    def receipt(self):
        return {'activity':list(self.rows),'actual_ACL_launch_count':sum(r['kind']=='FA.ACL_launch' for r in self.rows),
            'FA_wrapper_error_count':sum(r['kind']=='FA.wrapper' and not r['completed'] for r in self.rows),
            'original_fallback_policy_unchanged':True,'new_per_module_synchronization':False}

    def close(self):
        for owner,name,old,wrapper in reversed(self.saved):
            assert getattr(owner,name) is wrapper;setattr(owner,name,old)
        self.saved.clear()
