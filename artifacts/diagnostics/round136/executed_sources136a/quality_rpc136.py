"""Private all8 control for approved originalFalse/candidateTrue/restoredFalse49 runs.

No forward, encoder, VAE or sampler results are supplied by RPCs. The temporary
True flag covers the entire private pipeline, so this is a combination test,
not an isolated final-AdaLN causal quality claim. No adoption or tolerance.
"""
from pathlib import Path
import hashlib
import json
import os
import threading
from fingerprints129 import fingerprint,normalized_fingerprint
from candidate_adapter136 import FinalAdaLNCandidate136
from cohort128 import Cohort128,publish_fatal128

ROOT=Path('/tmp/h3_quality13601')
CASES={1:'original_false',2:'candidate_true',3:'restored_false'}


def runtime(torch):
    return {'pid':os.getpid(),'native_thread_id':threading.get_native_id(),'device':torch.npu.current_device(),
        'current_stream':int(torch.npu.current_stream().npu_stream),'deterministic':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32}


def master_state(model,torch):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    params={};empty=[]
    for name,param in model.named_parameters():
        local=param.to_local() if hasattr(param,'to_local') else param
        if local.numel():params[name]=fingerprint(local,torch,include_pointers=False)
        else:empty.append(name)
    for name,buffer in model.named_buffers():params['buffer:'+name]=fingerprint(buffer,torch,include_pointers=False)
    masters=[]
    for index,block in enumerate(model.blocks):
        hook=block._hook_registry._hooks[LayerwiseOffloadHook._HOOK_NAME]
        masters.append({'owner_block':index,'next_block':(index+1)%50,
            'cpu_master':{str(dtype):fingerprint(t,torch,include_pointers=True) for dtype,t in hook.dtype_cpu_flattened_weights.items()},
            'metadata_sha':hashlib.sha256(json.dumps({str(k):v for k,v in hook.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest(),
            'copy_stream':int(hook.copy_stream.npu_stream)})
    return {'materialized_parameters_and_buffers':params,'empty_parameter_names':empty,'native_masters':masters}


def publish(state):
    path=state['path'];path.write_text(json.dumps(state['report'],indent=2)+'\n')


def restore_local(state):
    errors=[];torch=state['torch']
    for label,operation in (('candidate',lambda:state['adapter'].restore() if state.get('adapter') is not None else None),
        ('initial_flags',lambda:torch.use_deterministic_algorithms(state['initial_flag'],warn_only=state['initial_warn'])),
        ('FA_observer',lambda:state['FA'].close() if state.get('FA') is not None else None)):
        try:operation()
        except BaseException as exc:errors.append({'operation':label,'type':type(exc).__name__,'message':str(exc)})
    return errors


def control(worker,action,index=0):
    import torch
    import torch_npu
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size,get_ulysses_parallel_world_size
    from fa_activity129 import FAActivity129
    rank=worker.rank;torch.npu.set_device(worker.device);model=worker.model_runner.pipeline.transformer
    state=getattr(model,'_quality136',None)
    if action=='prepare':
        assert state is None and not torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        assert len(model.blocks)==50 and get_tensor_model_parallel_world_size()==2 and get_sequence_parallel_world_size()==4 and get_ulysses_parallel_world_size()==4
        assert not torch.npu.matmul.allow_hf32 and not torch.npu.conv.allow_hf32
        assert all(int(b._hook_registry._hooks['layerwise_offload'].copy_stream.npu_stream)==int(torch.npu.current_stream().npu_stream) for b in model.blocks)
        path=ROOT/f'quality_worker_rank{rank}136.json';assert not path.exists()
        report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'TP_actual':2,'TP_group':list(get_tp_group().ranks),
            'SP_actual':4,'SP_group':list(get_sp_group().ranks),'initial_runtime':runtime(torch),'cases':[],
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},'candidate_adopted':False}
        state={'torch':torch,'initial_flag':False,'initial_warn':False,'path':path,'report':report,'cohort':Cohort128(torch,prefix='R136quality_control',timeout_s=240),
            'index':0,'case':None,'expected_flag':False,'candidate':False,'FA_count':0,'FA_errors':0,'adapter':None,'FA':None}
        model._quality136=state
    assert state is not None
    cohort=state['cohort'];report=state['report']
    try:
        if action=='prepare':
            report['model_before_state']=cohort.run('original_model_master_state_before',lambda:master_state(model,torch))
            def reference():
                original=json.loads(Path(f'/tmp/h3_repeat12902/repeat_rank{rank}129.json').read_text());assert original['completed']
                previous=original['model_before_state']['native_masters'];now=report['model_before_state']['native_masters'];assert len(previous)==len(now)==50
                for a,b in zip(now,previous):
                    assert a['owner_block']==b['owner_block'] and a['metadata_sha']==b['metadata_sha']
                    assert normalized_fingerprint(a['cpu_master'])==normalized_fingerprint(b['cpu_master'])
                return {'all50_original_R129_master_reference_SHA_verified':True}
            report['original_master_reference']=cohort.run('original_master_reference',reference)
            def install():
                state['adapter']=FinalAdaLNCandidate136(torch,model)
                state['FA']=FAActivity129(torch)
            cohort.run('transactional_prepare_adapter_and_FA',install)
            report['prepared']=True;cohort.run('publish_prepared',lambda:publish(state))
        elif action=='set':
            assert index in CASES and index==state['index']+1 and model._total_steps127==(index-1)*49
            def configure():
                torch.npu.synchronize()
                if index==2:state['adapter'].install()
                else:state['adapter'].restore();assert state['adapter'].physically_restored()
                torch.use_deterministic_algorithms(index==2,warn_only=False)
                state.update(index=index,case=CASES[index],expected_flag=index==2,candidate=index==2)
                assert torch.are_deterministic_algorithms_enabled()==state['expected_flag']
                return {'index':index,'case':CASES[index],'start_runtime':runtime(torch),'candidate_calls_before_request':state['adapter'].candidate.calls}
            row=cohort.run(f'configure_case{index}',configure);report['cases'].append(row);cohort.run(f'publish_case{index}',lambda:publish(state))
        elif action=='finish':
            assert state['index']==3 and model._total_steps127==147 and model._checks127==147*155
            def close():
                errors=restore_local(state);assert not errors,errors
                assert state['adapter'].physically_restored() and state['adapter'].candidate.calls==49
                assert state['adapter'].candidate.row_count_calls=={'2':1,'3':48}
                assert not torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            cohort.run('restore_all_private_methods_flags_FA',close)
            final=cohort.run('original_model_master_state_after',lambda:master_state(model,torch));report['model_after_state']=final
            assert final['native_masters']==report['model_before_state']['native_masters']
            assert normalized_fingerprint(final['materialized_parameters_and_buffers'])==normalized_fingerprint(report['model_before_state']['materialized_parameters_and_buffers'])
            report.update(completed=True,actual_original_evaluations=147,strict_Half_checks=model._checks127,
                actual_FA_launches=state['FA_count'],actual_FA_errors=state['FA_errors'],actual_candidate_calls=state['adapter'].candidate.calls,
                candidate_row_count_calls=state['adapter'].candidate.row_count_calls,candidate_contract=state['adapter'].candidate.last_contract,
                candidate_method_restored=True,final_runtime=runtime(torch),original_master_and_resident_state_unchanged=True)
            assert report['actual_FA_launches']==147*50 and report['actual_FA_errors']==0
            cohort.run('publish_finished',lambda:publish(state))
        elif action=='abort_restore':
            def abort_restore():
                errors=restore_local(state);assert not errors,errors
                assert state['adapter'].physically_restored() and not torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                return runtime(torch)
            report['aborted_runtime_after_restore']=cohort.run('aborted_request_physical_restore_only',abort_restore)
            report['aborted']=True;report['completed']=False;cohort.run('publish_aborted',lambda:publish(state))
        else:raise ValueError('Unknown private quality action')
        return {'rank':rank,'pid':os.getpid(),'action':action,'index':state['index'],'completed':report['completed'],'report':str(state['path'])}
    except BaseException as primary:
        report['failure']={'type':type(primary).__name__,'message':str(primary)}
        try:publish_fatal128({'ok':False,'rank':rank,'pid':os.getpid(),'phase':'quality_control_'+action,**report['failure']})
        except BaseException:pass
        report['local_cleanup_failures']=restore_local(state);report['completed']=False
        try:publish(state)
        except BaseException:pass
        raise


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r136_original_candidate_quality_control';assert not hasattr(DiffusionWorker,name)
    def method(self,action,index=0):
        try:return control(self,action,index)
        except BaseException as primary:
            # Includes imports, prepare assertions and store construction before
            # a state/cohort exists; never wait for peers before signalling fatal.
            try:publish_fatal128({'ok':False,'rank':self.rank,'pid':os.getpid(),'phase':'quality_RPC_entry_'+str(action),
                'index':index,'type':type(primary).__name__,'message':str(primary)})
            except BaseException:pass
            try:
                state=getattr(self.model_runner.pipeline.transformer,'_quality136',None)
                if state is not None:
                    errors=restore_local(state);state['report']['entry_cleanup_failures']=errors
                    state['report']['completed']=False
                    try:publish(state)
                    except BaseException:pass
            except BaseException:pass  # Metadata/cleanup must not replace primary.
            raise
    setattr(DiffusionWorker,name,method)
