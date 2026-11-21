"""Non-trajectory same-input replay on the actual formal worker thread.

Inputs come from this request's live NPU encoder/DiT invocation, never GPU/teacher
features. No replay output is fed into sampling. Original 155-check hooks remain
active in a separately counted benchmark scope. Only DiT50 native hooks change.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path('/tmp/h3_overlap12801')


def clone_tree(value,torch,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:clone_tree(v,torch,device) for k,v in value.items()}
    if isinstance(value,list):return [clone_tree(v,torch,device) for v in value]
    if isinstance(value,tuple):return tuple(clone_tree(v,torch,device) for v in value)
    assert value is None or isinstance(value,(str,int,float,bool)),('Unsupported forward input',type(value))
    return value


def master_sha(hooks,torch):
    return [{str(dtype):hashlib.sha256(memoryview(master.view(torch.uint8).numpy())).hexdigest()
             for dtype,master in hook.dtype_cpu_flattened_weights.items()} for hook in hooks]


def native_hooks(model):
    from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook
    owner_hooks=[]
    for block in model.blocks:
        registry=getattr(block,'_hook_registry',None);assert registry is not None
        hook=registry._hooks[LayerwiseOffloadHook._HOOK_NAME]
        owner_hooks.append(hook)
    # Native owner49->target0, owner0->target1, ...; preserve the actual ring links.
    hooks=owner_hooks[-1:]+owner_hooks[:-1]
    assert len(hooks)==50 and [id(h.next_block) for h in hooks]==[id(b) for b in model.blocks]
    return hooks


class MirrorLeafProof128:
    """Pair all104 original wide algorithms on SAME input and baseline-master weights.

    Return the original candidate result. Reference mirrors are temporary copies
    of immutable native host masters, not different quantization/math/inputs.
    No replacement output is used, and timing is explicitly nonclean diagnostic.
    """
    def __init__(self,torch,model,hooks):
        self.torch=torch;self.model=model;self.rows=[];self.saved=[]
        masters={}
        for index,hook in enumerate(hooks):
            for dtype,metadata in hook.dtype_metadata.items():
                backing=hook.dtype_cpu_flattened_weights[dtype]
                for row in metadata:
                    masters[f'blocks.{index}.{row["name"]}']=backing.as_strided(row['shape'],row['stride'],storage_offset=row['offset'])
        prepared=[]
        for name,layer in model.named_modules():
            old=getattr(layer,'quant_method',None)
            if type(old).__name__!='_MiniMaxH3HalfInputFP32OutputLinearMethod':continue
            reference=masters.get(name+'.weight')
            if reference is None:
                assert not name.startswith('blocks.'),'Missing exact native master mapping'
                reference=layer.weight.detach().cpu().clone()
            assert reference.dtype==torch.float16
            proof=self
            class Wrapped:
                def __init__(self,old,reference,name):self.old=old;self.reference=reference;self.name=name
                def __getattr__(self,key):return getattr(self.old,key)
                def apply(self,actual_layer,x,bias=None):
                    candidate=self.old.apply(actual_layer,x,bias)
                    mirror=self.reference.to(x.device)
                    original_layer=actual_layer
                    class ReadOnlyLayer:
                        weight=mirror
                        def __getattr__(self,key):return getattr(original_layer,key)
                    reference_result=self.old.apply(ReadOnlyLayer(),x,bias)
                    assert reference_result.shape==candidate.shape and reference_result.dtype==candidate.dtype==torch.float32
                    words=int((reference_result.contiguous().view(torch.int32)!=candidate.contiguous().view(torch.int32)).sum(dtype=torch.int32).cpu())
                    proof.rows.append({'module':self.name,'shape':list(candidate.shape),'elements':candidate.numel(),
                        'different_FP32_words':words,'bit_identical':words==0,'same_exact_leaf_input':True,
                        'original_algorithm_called_for_both':True,'original_candidate_result_returned':True})
                    return candidate
            wrapper=Wrapped(old,reference,name);prepared.append((layer,old,wrapper))
        assert len(prepared)==104
        try:
            for layer,old,wrapper in prepared:
                layer.quant_method=wrapper;self.saved.append((layer,old,wrapper))
        except BaseException:self.close();raise

    def close(self):
        for layer,old,wrapper in reversed(self.saved):
            assert layer.quant_method is wrapper;layer.quant_method=old
        self.saved.clear()


def benchmark_in_worker(model,kwargs,raw_forward,torch,event):
    import os
    from native_copy_adapter128 import NativeCopyAdapter128
    from cohort128 import Cohort128
    rank=torch.distributed.get_rank();device=torch.npu.current_device()
    cohort=Cohort128(torch);model._cohort128=cohort
    hooks=cohort.run('collect_native_ring',lambda:native_hooks(model))
    def tp_contract():
        from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size,get_tp_group
        size=get_tensor_model_parallel_world_size();ranks=list(get_tp_group().ranks)
        assert size==2 and len(ranks)==2 and int(model.blocks[0].attn.num_heads)==28
        return {'size':size,'physical_rank_group':ranks,'local_heads_before_USP4':28}
    actual_tp=cohort.run('actual_TP2_runtime_group',tp_contract)
    before_sha=cohort.run('master_before_SHA',lambda:master_sha(hooks,torch))
    before_dtype={str(p.dtype) for p in model.parameters()}
    cohort.run('validate_original_dtypes',lambda:None if before_dtype<={'torch.float16','torch.float32'} else (_ for _ in ()).throw(TypeError('Original dtype drift')))
    cpu_inputs=cohort.run('clone_live_input_CPU',lambda:clone_tree(kwargs,torch,'cpu'))
    inputs=cohort.run('clone_live_input_NPU',lambda:clone_tree(cpu_inputs,torch,f'npu:{device}'))
    path=ROOT/f'native_worker_input_step25_rank{rank}.pt'
    def save_input():
        assert not path.exists()
        with path.open('xb') as file:torch.save(cpu_inputs,file)
        return hashlib.sha256(path.read_bytes()).hexdigest()
    input_sha=cohort.run('save_native_input_fixture',save_input)
    report={'completed':False,'rank':rank,'pid':os.getpid(),'scope':__doc__,'input_source':'this formal request live NPU pipeline step25',
        'input_sha256':input_sha,'source_inputs_saved_only_on_NPU_host':True,'teacher_hidden_or_replay_outputs_used_by_sampler':False,
        'cases':[],'leaf_proof':[],'byte_proof':[],'profiles':[],'restored':False,
        'actual_model_parameter_dtypes':sorted(before_dtype),'actual_TP2_runtime':actual_tp,'original50blocks_TP2_USP4':True}
    receipt=ROOT/f'formal_replay_rank{rank}128.json'
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    def compare(actual,expected):
        assert actual.shape==expected.shape and actual.dtype==expected.dtype
        same=bool(torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)))
        delta=actual.double()-expected.double();denominator=float(expected.double().square().sum().sqrt().clamp_min(1e-30))
        return {'shape':list(actual.shape),'dtype':str(actual.dtype),'all_raw_words_equal':same,
            'max_abs_difference':float(delta.abs().max()),'relative_RMS_error':float(delta.square().sum().sqrt())/denominator}
    def all_ranks_true(value):
        tensor=torch.tensor([int(value)],dtype=torch.int32,device=f'npu:{device}')
        torch.distributed.all_reduce(tensor,op=torch.distributed.ReduceOp.MIN)
        return bool(int(tensor.cpu()))
    reference=None;active=None;all_equal=True
    def run(label,*,clean):
        nonlocal reference,all_equal
        cohort.run(label+'_drain_before_forward',torch.npu.synchronize);torch.distributed.barrier()
        before=model._benchmark_checks128;model._benchmark_active128=True
        def invoke():
            start=time.perf_counter();prediction=raw_forward(model,**inputs)
            actual=[x.detach().cpu().contiguous() for x in prediction]
            assert all(bool(torch.isfinite(x).all()) for x in actual)
            torch.npu.synchronize();elapsed=time.perf_counter()-start
            checks=model._benchmark_checks128-before;assert checks==155
            return actual,elapsed,checks
        try:actual,elapsed,checks=cohort.run(label+'_actual_forward',invoke)
        finally:model._benchmark_active128=False
        timer=torch.tensor([elapsed],dtype=torch.float32,device=f'npu:{device}')
        torch.distributed.all_reduce(timer,op=torch.distributed.ReduceOp.MAX)
        if reference is None:reference=[x.clone() for x in actual]
        compared=cohort.run(label+'_compare_outputs',lambda:[compare(x,y) for x,y in zip(actual,reference)])
        equal=all_ranks_true(all(x['all_raw_words_equal'] for x in compared));all_equal=all_equal and equal
        row={'mode':label,'clean':clean,'local_wall_includes_original_prediction_D2H_safety_s':elapsed,
            'rank_max_wall_s':float(timer.cpu()),'strict_Half_checks':checks,'full_output_leaves':compared,
            'all8ranks_all_output_words_equal_to_same_input_native':equal}
        report['cases'].append(row);cohort.run(label+'_publish_case',publish)
        event('R128_formal_replay_step',**row)
        return equal
    publish()
    try:
        for mode in ('native','serial','async','restored'):
            if active is not None:
                report['previous_adapter']=active.facts();cohort.run(mode+'_restore_prior_adapter',active.rollback);active=None
            if mode in ('serial','async'):
                holder={}
                def activate():holder['adapter']=NativeCopyAdapter128(torch,model.blocks,hooks,asynchronous=mode=='async').activate()
                try:cohort.run(mode+'_owner_thread_activate',activate)
                finally:active=holder.get('adapter')
                report['active_'+mode]=active.facts()
            if not run(mode+'_warmup',clean=False):break
            for repeat in range(3):
                if not run(mode,clean=True):break
            if not all_equal:break
            if active is not None:
                cohort.run(mode+'_profile_begin',active.begin_profile);run(mode+'_profile',clean=False)
                profile=cohort.run(mode+'_profile_end',active.end_profile)
                profile['actual_selected_hook_copy_stream']=int(hooks[0].copy_stream.npu_stream)
                profile['actual_selected_hook_copy_is_distinct']=hooks[0].copy_stream!=torch.npu.current_stream()
                profile['capability_copy_stream_is_unused_in_serial_control']=mode=='serial'
                report['profiles'].append(profile)
            if mode=='async':
                holder={};active.audit_each_copy=True
                def prepare_mirror():holder['proof']=MirrorLeafProof128(torch,model,hooks)
                try:
                    cohort.run('async_mirror104_prepare',prepare_mirror)
                    run('async_full_master_and_leaf_proof',clean=False)
                finally:
                    if 'proof' in holder:holder['proof'].close()
                    active.audit_each_copy=False
                proof=holder['proof']
                def validate_proof_structure():
                    assert len(proof.rows)==104
                    assert {r['target_block'] for r in active.byte_proofs}==set(range(50))
                cohort.run('async_proof_structure_and_mirror_restored',validate_proof_structure)
                report['leaf_proof']=proof.rows;report['byte_proof']=active.byte_proofs
                exact=all_ranks_true(all(r['bit_identical'] for r in proof.rows) and all(r['entire_native_flat_master_raw_words_equal'] for r in active.byte_proofs))
                all_equal=all_equal and exact
                if not exact:break
        if active is not None:cohort.run('restore_final_adapter',active.rollback);active=None
        # Always do a real restored output gate even if an earlier candidate failed numerically.
        restored=run('restored_final_gate',clean=False)
        after_sha=cohort.run('master_after_SHA',lambda:master_sha(hooks,torch))
        def validate_restored():
            assert after_sha==before_sha
            assert {str(p.dtype) for p in model.parameters()}==before_dtype
            assert all(h.copy_stream==torch.npu.current_stream() for h in hooks)
        cohort.run('all_master_dtype_and_native_streams_restored',validate_restored)
        report['all_cpu_master_SHA_unchanged']=True;report['restored']=restored
        report['warm_rank_max_medians_s']={m:statistics.median(r['rank_max_wall_s'] for r in report['cases'] if r['mode']==m and r['clean'])
            for m in ('native','serial','async','restored') if any(r['mode']==m and r['clean'] for r in report['cases'])}
        medians=report['warm_rank_max_medians_s'];baseline=medians.get('restored',medians['native']);candidate=medians.get('async')
        gain=(baseline-candidate)/baseline if candidate is not None else None
        report['candidate_gain_vs_restored_fraction']=gain
        report['all8ranks_bit_gates_passed']=all_equal
        report['candidate_formal_generation_allowed']=all_equal and gain is not None and gain>=0.01
        report['result']='Awaiting approval' if not all_equal else ('Keep for full validation' if report['candidate_formal_generation_allowed'] else 'Reject: no clear >=1% same-scope gain')
        report['completed']=True;cohort.run('persist_completed_gate_before_return_to_sampler',publish)
        event('R128_replay_gate_complete',**{k:v for k,v in report.items() if k not in ('cases','profiles','leaf_proof','byte_proof')})
        return report
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)}
        if active is not None:
            try:active.rollback();report['restored_hooks_after_exception']=True
            except BaseException as rollback:report['rollback_failure']={'type':type(rollback).__name__,'message':str(rollback)}
        publish();raise
