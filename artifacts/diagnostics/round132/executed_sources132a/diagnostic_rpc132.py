"""R132 original audio-tail first-change diagnostics; no new precision baseline.

Original50/155checks, TP2+USP4 and original FP32 exceptions stay. TemporaryTrue
is observational isolation only. Fixed full-input local head replays are not a
model/sampler activation cache. No49generation, no copy candidate or tolerance.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from diagnostic_state132 import clone,model_master_state,runtime_state,compare_time
from fingerprints129 import fingerprint,normalized_fingerprint,compare_outputs
from paired_words132 import compare_fp32

ROOT=Path('/tmp/h3_audio13201')
OLD=Path('/tmp/h3_overlap12801')


def diagnostic_rpc(worker):
    import torch
    import torch_npu
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context,get_forward_context
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size,get_ulysses_parallel_world_size,get_ring_parallel_world_size
    from h3_replay123.transformer import MiniMaxH3DiTModel as VerifiedDiT
    from audio_tail_trace132 import AudioTailTrace132
    from fa_activity129 import FAActivity129
    from cohort128 import Cohort128
    from observe_runtime127 import memory
    rank=worker.rank;torch.npu.set_device(worker.device);assert torch.npu.current_device()==rank
    model=worker.model_runner.pipeline.transformer
    initial_flag=torch.are_deterministic_algorithms_enabled();initial_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    assert not initial_flag and not initial_warn and len(model.blocks)==50
    assert get_tensor_model_parallel_world_size()==2 and get_sequence_parallel_world_size()==4 and get_ulysses_parallel_world_size()==4 and get_ring_parallel_world_size()==1
    assert model.blocks[0].attn.num_heads==28
    assert all(int(b._hook_registry._hooks['layerwise_offload'].copy_stream.npu_stream)==int(torch.npu.current_stream().npu_stream) for b in model.blocks)
    path=ROOT/f'audio_tail_rank{rank}132.json';assert not path.exists()
    report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'passes':[],
        'TP_actual':2,'TP_group':list(get_tp_group().ranks),'SP_actual':4,'SP_group':list(get_sp_group().ranks),'Ulysses_actual':4,'Ring_actual':1,
        'model_flag_not_adopted':True,'copy_candidate_enabled':False,'output_is_full50_prediction_not_decoded_media':True,
        'trace_scope':'Full and audio-row fingerprints; audio local input/weight/bias, preTP output and postTP output have complete word proofs'}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    cohort=Cohort128(torch,prefix='R132:audio_tail',timeout_s=360)
    def load():
        p=OLD/f'native_worker_input_step25_rank{rank}.pt';expected=json.loads((OLD/f'formal_replay_rank{rank}128.json').read_text())['input_sha256']
        with p.open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==expected
        report['source_input_sha256']=expected
        return torch.load(p,map_location='cpu',weights_only=True)
    cpu_input=cohort.run('load_own_input_SHA',load)
    before_state=cohort.run('master_and_resident_state_before',lambda:model_master_state(model,torch));report['model_before_state']=before_state
    def verify_masters():
        old=json.loads(Path(f'/tmp/h3_repeat12902/repeat_rank{rank}129.json').read_text())['model_before_state']['native_masters']
        assert len(old)==len(before_state['native_masters'])==50
        for current,reference in zip(before_state['native_masters'],old):
            assert current['owner_block']==reference['owner_block'] and current['metadata_sha']==reference['metadata_sha']
            assert normalized_fingerprint(current['cpu_master'])==normalized_fingerprint(reference['cpu_master'])
    cohort.run('verify_original_R129_master_values',verify_masters)
    time_holder={};time_handle=None;activity=None;trace=None;references={};time_references={};trace_first=None
    try:
        with set_current_vllm_config(worker.vllm_config),set_current_diffusion_config(worker.od_config),set_forward_context(vllm_config=worker.vllm_config,omni_diffusion_config=worker.od_config),torch.inference_mode():
            get_forward_context().sp_plan_hooks_applied=True
            inputs=cohort.run('clone_NPU_input_once',lambda:clone(cpu_input,torch,worker.device))
            audio_positions=VerifiedDiT._pos_ids(inputs['audio_pos_info'],'audio_pos_info').to(worker.device)
            holder={}
            def setup():
                def time_output(owner,args,result):
                    assert not time_holder;time_holder['time']=result.detach().clone()
                    return None
                holder['time']=model.time_embedder.register_forward_hook(time_output);holder['FA']=FAActivity129(torch)
            cohort.run('install_return_preserving_basic_observers',setup);time_handle=holder['time'];activity=holder['FA']
            plan=[('native_first',False,1),('true_control',True,3),('true_tail_trace',True,2),('true_restored',True,2),('native_final',False,1)]
            for phase,flag,repeats in plan:
                def flag_setup():
                    torch.npu.synchronize();torch.use_deterministic_algorithms(flag,warn_only=False)
                    assert torch.are_deterministic_algorithms_enabled()==flag
                cohort.run('set_'+phase+'_flag',flag_setup)
                if phase=='true_tail_trace':
                    holder={}
                    def install_trace():holder['trace']=AudioTailTrace132(torch,model,audio_positions,int(inputs['x'].shape[1]))
                    cohort.run('install_tail_original_return_observers',install_trace);trace=holder['trace']
                    report['tail_memory_budget']=trace.memory_budget;report['actual_local_dispatch']=trace.dispatch_info
                for repeat in range(repeats):
                    index=len(report['passes']);time_holder.clear();activity.reset()
                    if trace is not None:trace.reset()
                    input_before=cohort.run(f'input_before{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                    def forward():
                        torch.npu.synchronize();torch.npu.reset_peak_memory_stats();checks=model._checks127;started=time.perf_counter()
                        result=VerifiedDiT.forward(model,**inputs)
                        output=[v.detach().cpu().contiguous().clone() for v in result]
                        assert len(output)==2 and all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in output)
                        torch.npu.synchronize();elapsed=time.perf_counter()-started
                        count=model._checks127-checks;assert count==155
                        t=time_holder.pop('time').cpu().contiguous().clone();assert not time_holder
                        return output,t,elapsed,count,memory(torch)
                    output,t,elapsed,checks,mem=cohort.run(f'{phase}_full50_{repeat}',forward)
                    input_after=cohort.run(f'input_after{index}',lambda:fingerprint(inputs,torch,include_pointers=True))
                    assert normalized_fingerprint(input_before)==normalized_fingerprint(input_after)
                    if phase not in references:references[phase]=[v.clone() for v in output];time_references[phase]=t.clone()
                    fa=activity.receipt();assert fa['actual_ACL_launch_count']==50 and fa['FA_wrapper_error_count']==0
                    assert all(len(r['q_shape'])==4 and r['q_shape'][1]==7 for r in fa['activity'] if r['kind']=='FA.wrapper')
                    actual_flag=torch.are_deterministic_algorithms_enabled();assert actual_flag==flag
                    row={'phase':phase,'repeat':repeat,'strict_Half_checks':checks,'actual_deterministic_flag':actual_flag,
                        'full50_original_audit_D2H_and_observers_s':elapsed,'trace_enabled':trace is not None,'memory':mem,
                        'input_before':input_before,'input_after':input_after,'input_values_and_layout_unchanged':True,
                        'output_vs_phase_first':compare_outputs(output,references[phase],torch),
                        'output_vs_native_first':compare_outputs(output,references['native_first'],torch),
                        'time_vs_phase_first':compare_time(t,time_references[phase],torch),'actual_FA_activity':fa,'runtime':runtime_state(torch)}
                    if 'true_control' in references:row['output_vs_true_control_first']=compare_outputs(output,references['true_control'],torch)
                    report['passes'].append(row)
                    if trace is not None:
                        current=list(trace.rows);report.setdefault('tail_traces',[]).append(current)
                        if trace_first is None:trace_first=current
                        else:report['tail_trace_compare']=AudioTailTrace132.compare(trace_first,current)
                        report['paired_audio_local_apply']=trace.paired
                        report['paired_audio_post_TP_gather']=trace.paired_post_gather
                    cohort.run(f'publish_pass{index}',publish)
                if phase=='true_tail_trace':
                    cohort.run('remove_tail_hooks_before_component_replay',trace.close_hooks)
                    activity.reset()
                    def local_replay():
                        x,bias=trace.frozen;old=trace.original_audio_apply;layer=trace.audio_layer
                        assert layer.weight.shape==(16,5376) and layer.weight.dtype==torch.float32
                        assert bias is None or (bias.shape==(16,) and bias.dtype==torch.float32)
                        frozen_proof={'input':compare_fp32(x.detach().cpu().contiguous(),trace.references['x'],torch),
                            'weight':compare_fp32(layer.weight.detach().cpu().contiguous(),trace.references['weight'],torch),
                            'original_input_layout':trace.original_input_layout,
                            'frozen_input_layout':{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),
                                'storage_offset':x.storage_offset(),'device':str(x.device),'physical_NPU_format':'not_measured'}}
                        if bias is not None:frozen_proof['bias']=compare_fp32(bias.detach().cpu().contiguous(),trace.references['bias'],torch)
                        else:frozen_proof['bias']={'both_none':trace.references['bias'] is None}
                        assert frozen_proof['input']['all_words_equal'] and frozen_proof['weight']['all_words_equal']
                        assert frozen_proof['bias'].get('all_words_equal',frozen_proof['bias'].get('both_none',False))
                        x_fp=fingerprint(x,torch);w_fp=fingerprint(layer.weight,torch);b_fp=fingerprint(bias,torch)
                        checks=model._checks127;first=None;rows=[]
                        reference=trace.references['output']
                        for repeat in range(10):
                            result=old(layer,x,bias)
                            cpu=result.detach().cpu().contiguous().clone();torch.npu.synchronize()
                            if first is None:first=cpu.clone()
                            rows.append({'repeat':repeat,'full_output_vs_replay_first':compare_fp32(cpu,first,torch),
                                'audio_rows_vs_replay_first':compare_fp32(cpu.index_select(0,trace.positions_cpu),first.index_select(0,trace.positions_cpu),torch),
                                'full_output_vs_first_original_tail_call':compare_fp32(cpu,reference,torch)})
                        assert x_fp==fingerprint(x,torch) and w_fp==fingerprint(layer.weight,torch) and b_fp==fingerprint(bias,torch)
                        assert model._checks127==checks and activity.receipt()['actual_ACL_launch_count']==0
                        return {'scope':'10 original local audio apply calls on frozen FULL41792x5376 input; no module/TP gather, no model activation cache',
                            'same_frozen_input_original_parameter_and_bias_words':True,'original_vs_frozen_full_word_proof':frozen_proof,
                            'extra_full_model_checks':0,'actual_local_method_calls':10,'rows':rows,
                            'dispatcher':trace.dispatch_info['audio_out'],'input_shape':list(x.shape),'bias_present':bias is not None}
                    report['fixed_full_input_audio_local_replay']=cohort.run('original_local_audio_dispatch_repeats',local_replay)
                    cohort.run('release_all_tail_references',trace.close);trace=None
                    cohort.run('publish_component_replay',publish)
            cohort.run('remove_basic_time_observer',time_handle.remove);time_handle=None
            cohort.run('remove_FA_observer',activity.close);activity=None
            def restore():
                torch.npu.synchronize();torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
                assert not torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
            cohort.run('restore_original_false_flags',restore)
            after_state=cohort.run('master_and_resident_state_after',lambda:model_master_state(model,torch));report['model_after_state']=after_state
            assert before_state['native_masters']==after_state['native_masters']
            assert normalized_fingerprint(before_state['materialized_parameters_and_buffers'])==normalized_fingerprint(after_state['materialized_parameters_and_buffers'])
            report['original_master_and_resident_state_unchanged']=True;report['private_flags_physically_restored']=True
            report['final_runtime']=runtime_state(torch);report['completed']=True;cohort.run('publish_completed',publish)
            return {'rank':rank,'completed':True,'report':str(path),'full50_calls':9,'strict_checks':1395,'local_audio_calls':10,'flags_restored':True}
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)
        if trace is not None:trace.close()
        if time_handle is not None:time_handle.remove()
        if activity is not None:activity.close()


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r132_audio_first_change_diagnostic';assert not hasattr(DiffusionWorker,name)
    def method(self):return diagnostic_rpc(self)
    setattr(DiffusionWorker,name,method)
