"""Original fixed-input encoder and fixed-latent audio decoder localization only.

Native8 -> SHA-traced2 -> observer-restored4 calls for each component. Global
False/HF32off throughout; no replacement math, parametrization cache, candidate,
DiT forward or49-step sampler. All input/output tensors stay on the owned host.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import types
from fingerprints129 import fingerprint,normalized_fingerprint
from original_state137 import component_state,same_original_state,qualified
from cohort128 import Cohort128,publish_fatal128

ROOT=Path('/tmp/h3_components13703')
PRIOR=Path('/tmp/h3_quality13601/engine_quality136a/r01_original_false')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def clone(value,torch,device):
    if isinstance(value,torch.Tensor):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:clone(v,torch,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(clone(v,torch,device) for v in value)
    if isinstance(value,list):return [clone(v,torch,device) for v in value]
    return value


def input_word_pairs(actual,reference,torch,np):
    if isinstance(actual,torch.Tensor):
        assert isinstance(reference,torch.Tensor) and actual.shape==reference.shape and actual.dtype==reference.dtype
        a=actual.detach().cpu().contiguous();b=reference.detach().cpu().contiguous()
        width=actual.element_size();words={1:np.uint8,2:np.uint16,4:np.uint32,8:np.uint64}[width]
        av=a.reshape(-1).view(torch.uint8).numpy().view(words);bv=b.reshape(-1).view(torch.uint8).numpy().view(words)
        changed=int(np.count_nonzero(av!=bv));assert changed==0,'Original and fixed input words differ'
        return {'shape':list(actual.shape),'dtype':str(actual.dtype),'word_bits':width*8,'total_words':actual.numel(),
            'different_words':changed,'complete_word_range_equal':True}
    if isinstance(actual,dict):
        assert actual.keys()==reference.keys();return {k:input_word_pairs(v,reference[k],torch,np) for k,v in actual.items()}
    if isinstance(actual,(tuple,list)):
        assert isinstance(reference,type(actual)) and len(actual)==len(reference)
        return [input_word_pairs(a,b,torch,np) for a,b in zip(actual,reference)]
    assert actual==reference;return actual


def runtime(torch):
    from torch.nn.utils import parametrize
    return {'deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32,
        'current_stream':int(torch.npu.current_stream().npu_stream),'parametrization_cache_enabled':int(parametrize._cache_enabled),
        'cudnn_SDP_enabled':torch.backends.cuda.cudnn_sdp_enabled()}


def rng_state(torch):
    return {'CPU':fingerprint(torch.get_rng_state(),torch),'NPU':fingerprint(torch.npu.get_rng_state(),torch)}


def input_format(value,torch,torch_npu):
    if isinstance(value,torch.Tensor):
        if value.device.type!='npu':return {'device':str(value.device),'npu_format':None}
        try:return {'device':str(value.device),'npu_format':int(torch_npu.get_npu_format(value))}
        except Exception as exc:return {'device':str(value.device),'npu_format_unknown':type(exc).__name__}
    if isinstance(value,dict):return {k:input_format(v,torch,torch_npu) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [input_format(v,torch,torch_npu) for v in value]
    return None


def encoder_boundaries(encoder):
    points=[('encoder.text.embed_tokens',encoder.text_model.embed_tokens),('encoder.vision',encoder.vision),
        ('encoder.vision.patch_embed',encoder.vision.patch_embed)]
    points += [('encoder.vision.blocks.'+str(i),m) for i,m in enumerate(encoder.vision.blocks)]
    points += [('encoder.text',encoder.text_model)]
    points += [('encoder.text.layers.'+str(i),m) for i,m in enumerate(encoder.text_model.layers)]
    return points


def audio_boundaries(audio):
    model=audio.model;decoder=model.decoder
    points=[('audio.dec_in_proj',model.dec_in_proj),('audio.decoder',decoder),('audio.decoder.conv_pre',decoder.conv_pre)]
    points += [('audio.decoder.ups.'+str(i)+'.'+str(j),m) for i,ups in enumerate(decoder.ups) for j,m in enumerate(ups)]
    points += [('audio.decoder.resblocks.'+str(i),m) for i,m in enumerate(decoder.resblocks)]
    points += [('audio.decoder.activation_post',decoder.activation_post),('audio.decoder.conv_post',decoder.conv_post)]
    # Access only existing child modules, never dynamic module.weight properties.
    for name,module in decoder.named_modules():
        parametrizations=vars(module).get('_modules',{}).get('parametrizations')
        if parametrizations is not None and 'weight' in parametrizations:
            sequence=parametrizations['weight'];assert len(sequence)==1
            points.append(('audio.decoder.'+name+'.parametrizations.weight',sequence))
            points.append(('audio.decoder.'+name+'.parametrizations.weight.0',sequence[0]))
    return points


def diagnostic_rpc(worker):
    import torch
    import torch_npu
    import numpy as np
    from PIL import Image
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import prepare_encoder_inputs
    from npu_half_nan_compat124 import half_nan_scope
    from boundary_trace137 import BoundaryTrace137,first_boundary_change
    from quality_metrics136 import compare_float_arrays
    rank=worker.rank;torch.npu.set_device(worker.device);pipeline=worker.model_runner.pipeline;encoder=pipeline.text_encoder;audio=pipeline.audio_vae
    path=ROOT/f'component_rank{rank}137.json';assert not path.exists()
    report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'components':[],
        'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')},'DiT_forward_calls':0,'sampler_evaluations':0,
        'new_numeric_or_math_policy_adopted':False,'initial_runtime':runtime(torch)}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    cohort=Cohort128(torch,prefix='R137original_components',timeout_s=300)
    capture_installed=False;had_capture=False;capture_override=None;capture_binding=None;trace=None
    try:
        assert not report['initial_runtime']['deterministic'] and not report['initial_runtime']['warn_only']
        assert not report['initial_runtime']['HF32_matmul'] and not report['initial_runtime']['HF32_conv'] and not report['initial_runtime']['parametrization_cache_enabled']
        assert get_tensor_model_parallel_world_size()==2 and get_sequence_parallel_world_size()==4
        report['TP_actual']=2;report['TP_group']=list(get_tp_group().ranks);report['SP_actual']=4;report['SP_group']=list(get_sp_group().ranks)
        assert encoder.tp_size==8 and list(encoder.encoder_group.ranks)==list(range(8)) and len(encoder.text_model.layers)==50
        report['encoder_TP_group']=list(encoder.encoder_group.ranks);report['encoder_TP_rank']=encoder.encoder_group.rank_in_group
        # Original encoder inputs were not archived by R136: construct once through
        # the same original presentation/processor, then capture actual _encode args.
        previous=json.loads((PRIOR.parent/'lifecycle_result136.json').read_text());assert previous['completed']
        original=previous['requests'][0];assert original['label']=='r01_original_false'
        def prepare_rank0():
            if rank!=0:return None
            images=[]
            for name in ('first','last'):
                p=PRIOR/(name+'.png');assert sha(p)==original['images_sha256'][name]
                with Image.open(p) as image:assert image.mode=='RGB' and image.size==(1344,768);images.append(image.copy())
            sampling=OmniDiffusionSamplingParams(seed=1234,height=768,width=1344,num_frames=120,fps=24,frame_rate=24,
                num_inference_steps=49,extra_args={'task':'fl2va','duration_seconds':5,'audio_mode':'native','frame_indices':[0,-1]})
            raw={'prompt':'a red ball bouncing from left to right on the floor','multi_modal_data':{'image':images}}
            return prepare_encoder_inputs(raw,sampling,task='fl2va')
        with set_current_vllm_config(worker.vllm_config),set_current_diffusion_config(worker.od_config),set_forward_context(vllm_config=worker.vllm_config,omni_diffusion_config=worker.od_config),torch.inference_mode():
            prepared=cohort.run('native_presentation_prepare_rank0_only',prepare_rank0)
            fixed={};original_encode=encoder._encode;had_capture='_encode' in vars(encoder);capture_override=vars(encoder).get('_encode')
            def capture(self,*args,**kwargs):
                assert not fixed and len(args)==1 and args[0].ndim==1
                group=self.encoder_group;embedding=self.text_model.embed_tokens
                assert self.tp_size==8 and list(group.ranks)==list(range(8)) and embedding._tp_size==8
                assert embedding.weight.dtype==torch.float16 and embedding.weight.device.type=='npu'
                actual=(args,kwargs);input_before=fingerprint(actual,torch,include_pointers=True)
                fixed['args']=clone(args,torch,worker.device);fixed['kwargs']=clone(kwargs,torch,worker.device)
                fixed['fingerprint']=fingerprint((fixed['args'],fixed['kwargs']),torch,include_pointers=True)
                assert normalized_fingerprint(input_before)==normalized_fingerprint(fixed['fingerprint'])
                report['encoder_fixture']={'actual_input':input_before,'fixed_input':fixed['fingerprint'],
                    'original_vs_fixed_complete_word_pairs':input_word_pairs(actual,(fixed['args'],fixed['kwargs']),torch,np),
                    'original_format':input_format(actual,torch,torch_npu),'fixed_format':input_format((fixed['args'],fixed['kwargs']),torch,torch_npu),
                    'same_original_logical_input_values_and_layout':True,'vocab_shard_shape':list(embedding.weight.shape),
                    'vocab_shard_rank':embedding._tp_rank,'vocab_shard_device':str(embedding.weight.device),
                    'entry_method':original_encode.__func__.__module__+'.'+original_encode.__func__.__qualname__}
                if report['encoder_fixture']['original_format']!=report['encoder_fixture']['fixed_format']:raise RuntimeError('Fixed encoder clone changed available device format metadata')
                saved=ROOT/f'encoder_input_rank{rank}137.pt';assert not saved.exists()
                with saved.open('xb') as file:torch.save({'args':clone(args,torch,'cpu'),'kwargs':clone(kwargs,torch,'cpu'),'own_live_input':True},file)
                report['encoder_fixture']['host_input_archive_SHA']=sha(saved)
                output=original_encode(*args,**kwargs)
                assert normalized_fingerprint(input_before)==normalized_fingerprint(fingerprint(actual,torch,include_pointers=True))
                return output
            def install_capture():
                nonlocal capture_installed,capture_binding
                capture_binding=types.MethodType(capture,encoder);encoder._encode=capture_binding;capture_installed=True
            def restore_capture():
                nonlocal capture_installed
                if capture_installed:
                    assert vars(encoder).get('_encode') is capture_binding,'Do not overwrite an unexpected encoder method mutation'
                    if had_capture:encoder._encode=capture_override
                    else:delattr(encoder,'_encode')
                    capture_installed=False
            cohort.run('install_original_input_capture',install_capture)
            cohort.run('one_original_encoder_preparation_not_sampling',lambda:pipeline.encode_prompt(prepared))
            cohort.run('restore_original_encoder_method',restore_capture);assert fixed
            audio_files=list(PRIOR.glob('final_latents_pid*127.pt'));assert len(audio_files)==1 and sha(audio_files[0])==original['final_latents_sha256']
            audio_bundle=torch.load(audio_files[0],map_location='cpu',weights_only=True)
            assert audio_bundle['actual_DiT_evaluations']==49 and audio_bundle['audio_latent'].dtype==torch.float32 and audio_bundle['audio_latent'].shape==(2,32,207)
            latent=audio_bundle['audio_latent'].to(worker.device).clone();assert normalized_fingerprint(fingerprint(latent,torch))==normalized_fingerprint(fingerprint(audio_bundle['audio_latent'],torch))
            report['audio_fixture']={'source_file':str(audio_files[0]),'source_file_SHA':original['final_latents_sha256'],'fixed_latent':fingerprint(latent,torch,include_pointers=True),
                'original_cpu_vs_fixed_NPU_complete_word_pairs':input_word_pairs(audio_bundle['audio_latent'],latent,torch,np),
                'fixed_device_format':input_format(latent,torch,torch_npu),'original_native_FP32_latent_unchanged':True}
            for component_name,component,input_value,points in (
                ('encoder',encoder,(fixed['args'],fixed['kwargs']),encoder_boundaries(encoder)),
                ('audio_decoder',audio,latent,audio_boundaries(audio))):
                section={'component':component_name,'passes':[],'trace_first_change':None,'comparison_dtype':None,
                    'execution_parallel_scope':'Original encoder TP8' if component_name=='encoder' else 'Original full FP32 decoder independently replicated on each rank, not decoder TP8',
                    'all_flags_native_false':True,'observed_boundaries': [{'name':name,'class':qualified(module)} for name,module in points]}
                report['components'].append(section)
                section['state_before']=cohort.run(component_name+'_original_state_before',lambda:component_state(component,torch))
                first_native=None;traced_rows=[]
                for phase,count in (('native',8),('trace',2),('observer_restored',4)):
                    if phase=='trace':
                        def install_trace():
                            nonlocal trace
                            trace=BoundaryTrace137(torch,points)
                        cohort.run(component_name+'_install_return_preserving_trace',install_trace)
                    if phase=='observer_restored':
                        cohort.run(component_name+'_remove_own_trace_hooks',trace.close);trace=None
                    first_case=None
                    for repeat in range(count):
                        def call():
                            if trace is not None:trace.start_call(component_name+'_'+str(repeat))
                            before=fingerprint(input_value,torch,include_pointers=True);fmt_before=input_format(input_value,torch,torch_npu);rng_before=rng_state(torch)
                            assert not runtime(torch)['deterministic'] and not runtime(torch)['parametrization_cache_enabled']
                            torch.npu.synchronize();torch.npu.reset_peak_memory_stats();start=time.perf_counter()
                            if component_name=='encoder':
                                output=pipeline._encode_text_hidden(fixed['args'][0],fixed['kwargs']);compatibility=None
                            else:
                                assert all(p.dtype==torch.float32 for p in audio.parameters() if p.is_floating_point())
                                with pipeline._component_on_device(audio),half_nan_scope(enabled=True) as counter:
                                    output=audio.decode_latent(latent)
                                    output=output.detach().cpu().contiguous().clone();compatibility=dict(counter)
                            torch.npu.synchronize();elapsed=time.perf_counter()-start
                            cpu=output.detach().cpu().contiguous().clone();assert bool(torch.isfinite(cpu).all())
                            if component_name=='encoder':assert cpu.shape==(2043,5120) and cpu.dtype==torch.float16
                            else:assert cpu.shape==(1,2,165600) and cpu.dtype==torch.float32
                            after=fingerprint(input_value,torch,include_pointers=True);fmt_after=input_format(input_value,torch,torch_npu)
                            assert before==after and fmt_before==fmt_after
                            state=runtime(torch);assert not state['deterministic'] and not state['warn_only'] and not state['HF32_matmul'] and not state['HF32_conv'] and not state['parametrization_cache_enabled']
                            observed=trace.end_call() if trace is not None else None
                            return cpu,{'phase':phase,'repeat':repeat,'elapsed_full_component_sync_CPU_output_s':elapsed,
                                'input_before':before,'input_after':after,'input_values_layout_and_available_format_unchanged':True,
                                'output_SHA':fingerprint(cpu,torch)['sha256_logical_words'],'runtime':state,
                                'rng_before':rng_before,'rng_after':rng_state(torch),'compatibility_scope':compatibility,
                                'PyTorch_peak_allocated_GiB':torch.npu.max_memory_allocated()/(1024**3)},observed
                        cpu,row,observed=cohort.run(component_name+'_'+phase+'_'+str(repeat),call)
                        if first_native is None:first_native=cpu.clone()
                        if first_case is None:first_case=cpu.clone()
                        row['output_vs_phase_first']=compare_float_arrays(cpu.numpy(),first_case.numpy())
                        row['output_vs_native_first']=compare_float_arrays(cpu.numpy(),first_native.numpy())
                        section['comparison_dtype']=str(cpu.dtype);section['passes'].append(row)
                        if observed is not None:traced_rows.append(observed)
                        cohort.run(component_name+'_'+phase+'_'+str(repeat)+'_publish',publish)
                assert len(traced_rows)==2
                section['trace_rows']=traced_rows;section['trace_first_change']=first_boundary_change(traced_rows[0],traced_rows[1])
                section['state_after']=cohort.run(component_name+'_original_state_after',lambda:component_state(component,torch))
                assert same_original_state(section['state_before'],section['state_after']), 'Original component parameters/buffers/CPU masters changed'
                section['original_parameters_buffers_and_CPU_masters_unchanged']=True;section['own_observers_removed']=trace is None
                cohort.run(component_name+'_completed_publish',publish)
            report['final_runtime']=runtime(torch);assert report['final_runtime']==report['initial_runtime']
            report['original_encoder_method_restored']=('_encode' in vars(encoder))==had_capture and (not had_capture or vars(encoder)['_encode'] is capture_override)
            assert report['original_encoder_method_restored'] and encoder._encode.__func__ is original_encode.__func__;report['completed']=True;cohort.run('all8_original_component_diagnostic_complete',publish)
            return {'rank':rank,'completed':True,'report':str(path),'encoder_fixed_repeats':14,'audio_fixed_repeats':14,'new_math':False}
    except BaseException as primary:
        report['failure']={'type':type(primary).__name__,'message':str(primary)};report['completed']=False
        try:publish_fatal128({'ok':False,'rank':rank,'pid':os.getpid(),'phase':'original_component_RPC',**report['failure']})
        except BaseException:pass
        try:publish()
        except BaseException:pass
        raise
    finally:
        import sys
        primary=sys.exc_info()[1];errors=[]
        if capture_installed:
            try:
                assert vars(encoder).get('_encode') is capture_binding,'Do not overwrite an unexpected encoder method mutation'
                if had_capture:encoder._encode=capture_override
                else:delattr(encoder,'_encode')
            except BaseException as exc:errors.append({'operation':'capture_restore','message':str(exc)})
        if trace is not None:
            try:trace.close()
            except BaseException as exc:errors.append({'operation':'trace_remove','message':str(exc)})
        if errors:
            report['cleanup_failures']=errors;report['cleanup_state_unknown']=True;report['completed']=False
            try:publish()
            except BaseException:pass
            try:publish_fatal128({'ok':False,'rank':rank,'pid':os.getpid(),'phase':'original_component_cleanup','errors':errors})
            except BaseException:pass
            if primary is None:raise RuntimeError('Original component cleanup incomplete: '+json.dumps(errors))


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r137_original_components_fixed_repeat';assert not hasattr(DiffusionWorker,name)
    def method(self):
        try:return diagnostic_rpc(self)
        except BaseException as primary:
            try:publish_fatal128({'ok':False,'rank':self.rank,'pid':os.getpid(),'phase':'component_RPC_entry','type':type(primary).__name__,'message':str(primary)})
            except BaseException:pass
            raise
    setattr(DiffusionWorker,name,method)
