"""Authorized full50 same-input diagnostic: native->deterministic->restored only.

No sampler/generation or copy candidate. Process-wide deterministic flag changes
are diagnostic, not adopted precision policy. Original FP32 exceptions/155checks
and actual TP2+USP4 remain. All tensor snapshots/weights stay on the owned host.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from fingerprints129 import fingerprint,normalized_fingerprint,compare_outputs

ROOT=Path('/tmp/h3_full5013101')
OLD=Path('/tmp/h3_overlap12801')


def clone(value,torch,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:clone(v,torch,device) for k,v in value.items()}
    if isinstance(value,list):return [clone(v,torch,device) for v in value]
    if isinstance(value,tuple):return tuple(clone(v,torch,device) for v in value)
    return value


def model_master_state(model,torch):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    rows={};offloaded=[]
    for name,param in model.named_parameters():
        local=param.to_local() if hasattr(param,'to_local') else param
        if local.numel():rows[name]=fingerprint(local,torch,include_pointers=False)
        else:offloaded.append(name)
    for name,buffer in model.named_buffers():rows['buffer:'+name]=fingerprint(buffer,torch,include_pointers=False)
    masters=[]
    for index,block in enumerate(model.blocks):
        hook=block._hook_registry._hooks[LayerwiseOffloadHook._HOOK_NAME]
        masters.append({'owner_block':index,'next_block':(index+1)%50,'cpu_master':{
            str(dtype):fingerprint(t,torch,include_pointers=True) for dtype,t in hook.dtype_cpu_flattened_weights.items()},
            'metadata_sha':hashlib.sha256(json.dumps({str(k):v for k,v in hook.dtype_metadata.items()},sort_keys=True,default=str).encode()).hexdigest(),
            'copy_stream':int(hook.copy_stream.npu_stream)})
    return {'materialized_parameters_and_buffers':rows,'empty_parameter_names':offloaded,'native_masters':masters}


def runtime_state(torch):
    return {'pid':os.getpid(),'native_thread_id':threading.get_native_id(),'device':torch.npu.current_device(),
        'current_stream':int(torch.npu.current_stream().npu_stream),
        'deterministic':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32}


def compare_time(actual,reference,torch):
    assert actual.shape==reference.shape and actual.dtype==reference.dtype==torch.float32
    changed=int((actual.reshape(-1).view(torch.int32)!=reference.reshape(-1).view(torch.int32)).sum())
    delta=actual.double()-reference.double();den=float(reference.double().square().sum().sqrt().clamp_min(1e-30))
    return {'shape':list(actual.shape),'dtype':str(actual.dtype),'different_FP32_words':changed,'all_words_equal':changed==0,
        'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/den}


def diagnostic_rpc(worker):
    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context,get_forward_context
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size,get_ulysses_parallel_world_size,get_ring_parallel_world_size
    from h3_replay123.transformer import MiniMaxH3DiTModel as VerifiedDiT
    from cohort128 import Cohort128
    from fa_activity129 import FAActivity129
    from observe_runtime127 import memory
    rank=worker.rank;device=worker.device;torch.npu.set_device(device)
    assert torch.npu.current_device()==rank
    model=worker.model_runner.pipeline.transformer
    initial_flag=torch.are_deterministic_algorithms_enabled();initial_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    assert not initial_flag and not initial_warn,'Require the original false deterministic baseline'
    assert len(model.blocks)==50 and get_tensor_model_parallel_world_size()==2
    assert model.blocks[0].attn.num_heads==28
    assert get_sequence_parallel_world_size()==4 and get_ulysses_parallel_world_size()==4 and get_ring_parallel_world_size()==1
    assert not torch.npu.matmul.allow_hf32 and not torch.npu.conv.allow_hf32
    path=ROOT/f'full50_rank{rank}131.json';assert not path.exists()
    report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'passes':[],
        'TP_actual':get_tensor_model_parallel_world_size(),'TP_group':list(get_tp_group().ranks),'actual_local_heads':28,
        'SP_actual':get_sequence_parallel_world_size(),'SP_group':list(get_sp_group().ranks),
        'Ulysses_actual':get_ulysses_parallel_world_size(),'Ring_actual':get_ring_parallel_world_size(),
        'initial_deterministic_flag':initial_flag,'initial_warn_only':initial_warn,
        'copy_candidate_enabled':False,'new_precision_policy_adopted':False,'sampler_steps':0,
        'flag_scope':'Entire private process during each case, not just the time projection',
        'output_scope':'Original full50 forward video/audio prediction tensors on own-live step25 input, not decoded49-step media',
        'time_capture_scope':'One same-current-stream output clone per full50 call, no intermediate D2H; nonclean diagnostic'}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    cohort=Cohort128(torch,prefix='R131:full50',timeout_s=240)
    assert all(int(block._hook_registry._hooks['layerwise_offload'].copy_stream.npu_stream)==int(torch.npu.current_stream().npu_stream) for block in model.blocks),'Keep original shared-copy baseline'
    input_path=OLD/f'native_worker_input_step25_rank{rank}.pt'
    expected=json.loads((OLD/f'formal_replay_rank{rank}128.json').read_text())['input_sha256']
    def load():
        with input_path.open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==expected
        return torch.load(input_path,map_location='cpu',weights_only=True)
    cpu_input=cohort.run('load_own_native_input_SHA',load)
    report['source_input_sha256']=expected
    initial=cohort.run('full_native_master_and_resident_state_before',lambda:model_master_state(model,torch));report['model_before_state']=initial
    def verify_original_masters():
        reference_path=Path(f'/tmp/h3_repeat12902/repeat_rank{rank}129.json')
        previous=json.loads(reference_path.read_text());assert previous['completed']
        old_masters=previous['model_before_state']['native_masters'];assert len(old_masters)==len(initial['native_masters'])==50
        for current,reference in zip(initial['native_masters'],old_masters):
            assert current['owner_block']==reference['owner_block'] and current['metadata_sha']==reference['metadata_sha']
            assert normalized_fingerprint(current['cpu_master'])==normalized_fingerprint(reference['cpu_master']), 'Original R129 CPU master values changed'
        return {'reference':str(reference_path),'all50_original_R129_master_logical_words_and_metadata_same':True}
    report['original_master_reference_verified']=cohort.run('verify_original_R129_master_SHA_reference',verify_original_masters)
    time_handle=None;activity=None;time_holder={};video_audio_refs={};time_refs={}
    try:
        with set_current_vllm_config(worker.vllm_config),set_current_diffusion_config(worker.od_config),set_forward_context(vllm_config=worker.vllm_config,omni_diffusion_config=worker.od_config),torch.inference_mode():
            get_forward_context().sp_plan_hooks_applied=True
            inputs=cohort.run('clone_input_NPU_once',lambda:clone(cpu_input,torch,device))
            def capture_time(module,args,output):
                assert 'time' not in time_holder,'Exactly one time embedder call per original graph'
                assert output.dtype==torch.float32
                time_holder['time']=output.detach().clone()
                return None  # Preserve original module return, clone never participates in math.
            holder={}
            def install_observers():
                holder['time']=model.time_embedder.register_forward_hook(capture_time);holder['FA']=FAActivity129(torch)
            cohort.run('install_same_allrank_time_and_FA_observers',install_observers)
            time_handle=holder['time'];activity=holder['FA']
            for case,flag in [('native',False),('deterministic',True),('restored',False)]:
                def set_flag():
                    torch.npu.synchronize();torch.use_deterministic_algorithms(flag,warn_only=False)
                    assert torch.are_deterministic_algorithms_enabled()==flag
                    return runtime_state(torch)
                report.setdefault('case_start_runtime',{})[case]=cohort.run('set_'+case+'_flag_all8_before_forward',set_flag)
                for repeat in range(3):
                    index=len(report['passes']);time_holder.clear();activity.reset()
                    before=cohort.run(f'input_before{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                    def forward():
                        torch.npu.synchronize();torch.npu.reset_peak_memory_stats();checks=model._checks127;started=time.perf_counter()
                        output=VerifiedDiT.forward(model,**inputs)
                        actual=[v.detach().cpu().contiguous().clone() for v in output]
                        assert len(actual)==2 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in actual)
                        torch.npu.synchronize();elapsed=time.perf_counter()-started
                        count=model._checks127-checks;assert count==155
                        assert 'time' in time_holder
                        time_cpu=time_holder.pop('time').detach().cpu().contiguous().clone()
                        assert bool(torch.isfinite(time_cpu).all())
                        return actual,time_cpu,elapsed,count,memory(torch)
                    output,time_output,elapsed,checks,mem=cohort.run(f'{case}_original50forward{repeat}',forward)
                    after=cohort.run(f'input_after{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                    assert normalized_fingerprint(before)==normalized_fingerprint(after),'Diagnostic input changed'
                    if case not in video_audio_refs:
                        video_audio_refs[case]=[v.clone() for v in output];time_refs[case]=time_output.clone()
                    if case=='native':
                        video_audio_refs['native_latest']=[v.clone() for v in output];time_refs['native_latest']=time_output.clone()
                    fa=activity.receipt();assert fa['actual_ACL_launch_count']==50 and fa['FA_wrapper_error_count']==0
                    wrapper_shapes=[r['q_shape'] for r in fa['activity'] if r['kind']=='FA.wrapper']
                    assert len(wrapper_shapes)==50 and all(len(shape)==4 and shape[1]==7 for shape in wrapper_shapes), 'Actual Ulysses attention head partition differs from TP2/USP4'
                    row={'case':case,'repeat':repeat,'strict_Half_checks':checks,
                        'actual_deterministic_flag':torch.are_deterministic_algorithms_enabled(),
                        'local_full50_including_original_audit_and_output_D2H_s':elapsed,'memory':mem,
                        'input_before':before,'input_after':after,'input_values_and_layout_pre_post_equal':True,
                        'output_vs_same_case_first':compare_outputs(output,video_audio_refs[case],torch),
                        'output_vs_native_first':compare_outputs(output,video_audio_refs['native'],torch),
                        'output_vs_native_last':compare_outputs(output,video_audio_refs['native_latest'],torch) if case!='native' else None,
                        'time_vs_same_case_first':compare_time(time_output,time_refs[case],torch),
                        'time_vs_native_first':compare_time(time_output,time_refs['native'],torch),
                        'time_vs_native_last':compare_time(time_output,time_refs['native_latest'],torch) if case!='native' else None,
                        'native_last_anchor_scope':'Final native repeat2, applicable only to deterministic/restored cases',
                        'actual_FA_activity':fa,'runtime':runtime_state(torch)}
                    if 'deterministic' in video_audio_refs:
                        row['output_vs_deterministic_first']=compare_outputs(output,video_audio_refs['deterministic'],torch)
                    report['passes'].append(row);cohort.run(f'publish_pass{index}',publish)
            cohort.run('remove_time_output_observer',time_handle.remove);time_handle=None
            cohort.run('remove_FA_observer',activity.close);activity=None
            def restore():
                torch.npu.synchronize();torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
                assert torch.are_deterministic_algorithms_enabled()==initial_flag
                assert torch.is_deterministic_algorithms_warn_only_enabled()==initial_warn
            cohort.run('restore_initial_private_flags',restore)
            final=cohort.run('full_native_master_and_resident_state_after',lambda:model_master_state(model,torch));report['model_after_state']=final
            report['native_CPU_masters_words_metadata_unchanged']=initial['native_masters']==final['native_masters']
            report['materialized_parameters_buffers_unchanged']=normalized_fingerprint(initial['materialized_parameters_and_buffers'])==normalized_fingerprint(final['materialized_parameters_and_buffers'])
            assert report['native_CPU_masters_words_metadata_unchanged'] and report['materialized_parameters_buffers_unchanged']
            report['private_flags_physically_restored']=True;report['final_runtime']=runtime_state(torch)
            report['completed']=True;cohort.run('publish_completed_before_driver_shutdown',publish)
            return {'rank':rank,'completed':True,'report':str(path),'original50forwards':9,'strict_checks':9*155,
                'physical_flags_restored':True,'candidate_policy_adopted':False}
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};report['flag_at_failure']=torch.are_deterministic_algorithms_enabled();publish();raise
    finally:
        torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
        if time_handle is not None:time_handle.remove()
        if activity is not None:activity.close()


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r131_full50_deterministic_diagnostic';assert not hasattr(DiffusionWorker,name)
    def method(self):return diagnostic_rpc(self)
    setattr(DiffusionWorker,name,method)
