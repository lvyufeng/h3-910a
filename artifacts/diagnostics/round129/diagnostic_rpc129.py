"""Actual formal-worker read-only repeatability RPC; no sampling or new copy-stream.

Uses prior own-live NPU inputs from host. Every diagnostic invocation is the
original50-block graph and155strict checks, not a generation step or acceptance.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from fingerprints129 import fingerprint,normalized_fingerprint,digest,cpu_tensor_sha,compare_outputs

ROOT=Path('/tmp/h3_repeat12902')
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
    from vllm_omni.diffusion.attention.backends import fa910a
    def tensors(cache):
        result=[]
        for key,value in cache.items():
            values=value if isinstance(value,(list,tuple)) else (value,)
            result.append({'key':repr(key),'tensors':[{'shape':list(v.shape),'dtype':str(v.dtype),'pointer':v.data_ptr(),'stride':list(v.stride())}
                for v in values if isinstance(v,torch.Tensor)]})
        return result
    return {'pid':os.getpid(),'native_thread_id':threading.get_native_id(),'device':torch.npu.current_device(),
        'current_stream':int(torch.npu.current_stream().npu_stream),'FA_scratch':tensors(getattr(fa910a,'_SCRATCH',{})),
        'FA_pad_buffers':tensors(getattr(fa910a,'_PAD_BUFFERS',{})),
        'FA_kernel_keys':[repr(k) for k in getattr(fa910a,'_CACHE',{})]}


def context_state129():
    from vllm_omni.diffusion.forward_context import get_forward_context
    context=get_forward_context()
    return {name:repr(value) for name,value in vars(context).items() if name not in ('omni_diffusion_config','vllm_config')}


def diagnostic_rpc(worker,mode):
    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context,get_forward_context
    from h3_replay123.transformer import MiniMaxH3DiTModel as VerifiedDiT
    from cohort128 import Cohort128
    from region_trace129 import RegionTrace129
    rank=worker.rank;device=worker.device
    torch.npu.set_device(device);assert torch.npu.current_device()==rank
    assert mode in ('repeat','trace','detailed')
    model=worker.model_runner.pipeline.transformer
    report={'scope':__doc__,'mode':mode,'rank':rank,'pid':os.getpid(),'completed':False,'passes':[],
        'copy_stream_candidate_enabled':False,'model_inference_not_sampler_steps':True,'original_F32_exceptions':True}
    path=ROOT/f'{mode}_rank{rank}129.json';assert not path.exists()
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    cohort=Cohort128(torch,prefix='R129:'+mode,timeout_s=120)
    # Preserve false-gate numeric outcomes; true runtime exceptions abort the private cohort.
    assert all(int(block._hook_registry._hooks['layerwise_offload'].copy_stream.npu_stream)==int(torch.npu.current_stream().npu_stream) for block in model.blocks), 'Native shared-copy baseline required'
    input_path=OLD/f'native_worker_input_step25_rank{rank}.pt'
    expected=json.loads((OLD/f'formal_replay_rank{rank}128.json').read_text())['input_sha256']
    def load():
        with input_path.open('rb') as file:actual=hashlib.file_digest(file,'sha256').hexdigest()
        assert actual==expected
        return torch.load(input_path,map_location='cpu',weights_only=True)
    cpu_input=cohort.run('load_own_native_input_SHA',load)
    report['source_input_sha256']=expected;report['initial_runtime_state']=runtime_state(torch)
    initial=cohort.run('full_native_master_and_resident_state_before',lambda:model_master_state(model,torch))
    report['model_before_state']=initial
    trace=None;trace_reference=None;reference=None;activity=None
    try:
        with set_current_vllm_config(worker.vllm_config),set_current_diffusion_config(worker.od_config),set_forward_context(vllm_config=worker.vllm_config,omni_diffusion_config=worker.od_config),torch.inference_mode():
            get_forward_context().sp_plan_hooks_applied=True
            from fa_activity129 import FAActivity129
            holder={}
            def install_activity():holder['activity']=FAActivity129(torch)
            cohort.run('install_FA_activity_counter',install_activity);activity=holder['activity']
            inputs=cohort.run('clone_input_NPU',lambda:clone(cpu_input,torch,device))
            if mode!='repeat':
                detailed=None
                if mode=='detailed':
                    earlier=json.loads((ROOT/f'trace_rank{rank}129.json').read_text())
                    first=earlier['trace_compare'][0]['first_different_trace_boundary']
                    if first and first['module'].startswith('blocks.'):
                        detailed=int(first['module'].split('.')[1])
                holder={}
                def install_trace():holder['trace']=RegionTrace129(torch,model,detailed_block=detailed)
                cohort.run('install_trace_same_allrank_graph',install_trace);trace=holder['trace']
                report['detailed_block']=detailed
            report['trace_compare']=[]
            for index in range(3 if mode=='repeat' else 2):
                # Compare same allocation reuse first, then a fresh identical input allocation.
                if mode=='repeat' and index==2:inputs=cohort.run('fresh_input_allocation',lambda:clone(cpu_input,torch,device))
                context_before=context_state129()
                before=cohort.run(f'input_before{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                if trace is not None:trace.reset()
                activity.reset()
                def forward():
                    torch.npu.synchronize();start=time.perf_counter();checks=model._checks127
                    output=VerifiedDiT.forward(model,**inputs)
                    actual=[v.detach().cpu().contiguous() for v in output]
                    assert all(bool(torch.isfinite(v).all()) for v in actual)
                    torch.npu.synchronize();elapsed=time.perf_counter()-start
                    count=model._checks127-checks;assert count==155
                    return actual,elapsed,count
                output,elapsed,checks=cohort.run(f'original50forward{index}',forward)
                after=cohort.run(f'input_after{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                input_equal=normalized_fingerprint(before)==normalized_fingerprint(after)
                if reference is None:reference=[v.clone() for v in output]
                compared=compare_outputs(output,reference,torch)
                row={'pass':index,'local_full_forward_and_original_D2H_audit_s':elapsed,'strict_Half_checks':checks,
                    'input_values_and_layout_pre_post_equal':input_equal,'input_before':before,'input_after':after,
                    'same_input_fresh_allocation':mode=='repeat' and index==2,'output_comparison_to_pass0':compared,
                    'runtime':runtime_state(torch),'context_before':context_before,'context_after':context_state129(),
                    'leaf_trace_synchronization_enabled':trace is not None,'actual_FA_activity':activity.receipt()}
                report['passes'].append(row)
                if trace is not None:
                    current=list(trace.rows);report.setdefault('traces',[]).append(current)
                    if trace_reference is None:trace_reference=current
                    else:report['trace_compare'].append(RegionTrace129.compare(trace_reference,current))
                cohort.run(f'publish_pass{index}',publish)
            if trace is not None:cohort.run('remove_all_trace_observers',trace.close);trace=None
            cohort.run('remove_FA_activity_counter',activity.close);activity=None
            final=cohort.run('full_native_master_and_resident_state_after',lambda:model_master_state(model,torch))
            report['model_after_state']=final
            report['all_native_CPU_masters_metadata_and_words_same']=initial['native_masters']==final['native_masters']
            report['all_materialized_resident_parameters_buffers_same']=normalized_fingerprint(initial['materialized_parameters_and_buffers'])==normalized_fingerprint(final['materialized_parameters_and_buffers'])
            report['completed']=True;report['final_runtime_state']=runtime_state(torch)
            cohort.run('publish_completed_before_next_RPC',publish)
            return {'rank':rank,'completed':True,'report':str(path),'original50forward_count':len(report['passes']),
                'checks':155*len(report['passes']),'first_trace_change':report['trace_compare'][0]['first_different_trace_boundary'] if report['trace_compare'] else None}
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if trace is not None:trace.close()
        if activity is not None:activity.close()


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r129_native_repeatability'
    assert not hasattr(DiffusionWorker,name)
    def method(self,mode):return diagnostic_rpc(self,mode)
    setattr(DiffusionWorker,name,method)
