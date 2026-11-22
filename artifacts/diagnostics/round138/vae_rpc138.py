"""Measure original video-VAE decode timing and bitwise repeatability only.

Fixed R136 original-True latent, original FP32 VAE parameters, no math change,
no candidate, no sampler. Four decodes per rank: repeat 1 is staging-cold and
reported separately; repeats 2-4 are warm. Outputs are compared on the complete
FP32 word domain. Timing is decomposed at the public component boundary.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from fingerprints129 import fingerprint, normalized_fingerprint
from original_state137 import component_state, same_original_state

ROOT=Path('/tmp/h3_vae13803')
PRIOR=Path('/tmp/h3_quality13601/engine_quality136a/r01_original_false')
REPEATS=4


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def runtime(torch):
    from torch.nn.utils import parametrize
    return {'deterministic':torch.are_deterministic_algorithms_enabled(),
        'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'HF32_matmul':torch.npu.matmul.allow_hf32,'HF32_conv':torch.npu.conv.allow_hf32,
        'current_stream':int(torch.npu.current_stream().npu_stream),
        'parametrization_cache_enabled':int(parametrize._cache_enabled),
        'cudnn_SDP_enabled':torch.backends.cuda.cudnn_sdp_enabled()}


def rng_state(torch):
    return {'CPU':fingerprint(torch.get_rng_state(),torch),'NPU':fingerprint(torch.npu.get_rng_state(),torch)}


def word_compare(a,b,torch,np):
    a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous()
    assert a.shape==b.shape and a.dtype==b.dtype
    width=a.element_size();words={1:np.uint8,2:np.uint16,4:np.uint32,8:np.uint64}[width]
    av=a.reshape(-1).view(torch.uint8).numpy().view(words);bv=b.reshape(-1).view(torch.uint8).numpy().view(words)
    changed=int(np.count_nonzero(av!=bv));total=int(av.size)
    if changed:
        diff=np.abs(av.astype(np.float64)-bv.astype(np.float64))
        rms=float(np.sqrt(np.mean((av.astype(np.float64)-bv.astype(np.float64))**2)))
        ref=float(np.sqrt(np.mean(bv.astype(np.float64)**2)))
        maxabs=float(diff.max())
        maxrel=(maxabs/ref*100.0) if ref else None
        return {'word_bits':width*8,'total_words':total,'different_words':changed,'all_words_equal':False,
            'max_abs_difference':maxabs,'relative_RMS_percent':(float(rms/ref*100.0) if ref else None),
            'max_abs_over_reference_RMS_percent':maxrel}
    return {'word_bits':width*8,'total_words':total,'different_words':0,'all_words_equal':True,
        'max_abs_difference':0.0,'relative_RMS_percent':0.0,'max_abs_over_reference_RMS_percent':0.0}


def tiling_facts(video_vae,latent,torch):
    model=video_vae.model
    facts={'parallel_size':int(getattr(video_vae,'parallel_size',1)),'distributed_enabled':bool(video_vae.is_distributed_enabled())}
    ratio=int(model.vae_ratio);facts['vae_ratio']=ratio
    height=int(latent.shape[-2])*ratio;width=int(latent.shape[-1])*ratio
    facts['pixel_height']=height;facts['pixel_width']=width
    rows,row_len,row_overlap=model.split_tiles(height,True)
    cols,col_len,col_overlap=model.split_tiles(width,True)
    facts.update(tile_rows=len(rows),tile_cols=len(cols),num_tiles=len(rows)*len(cols),
        row_positions=[int(v) for v in rows],row_lengths=[int(v) for v in row_len],row_overlaps=[int(v) for v in row_overlap],
        col_positions=[int(v) for v in cols],col_lengths=[int(v) for v in col_len],col_overlaps=[int(v) for v in col_overlap],
        decoder_tile_count=int(video_vae._decoder_tile_count(latent)))
    sp_size=facts['parallel_size']
    if sp_size>1:
        state=video_vae._native_parallel_state()
        rank=int(state['sp_rank']);size=int(state['sp_size'])
        local=model._local_tile_indices(facts['num_tiles'],rank,size)
        group=state['sp_process_group']
        # Portable process-group introspection: this torch build has no
        # ProcessGroup.ranks attribute, and tile ownership only needs size/rank.
        ranks=None
        getter=getattr(torch.distributed,'get_process_group_ranks',None)
        if callable(getter):
            try:ranks=[int(v) for v in getter(group)]
            except BaseException:ranks=None
        world=int(torch.distributed.get_world_size(group))
        facts.update(sp_group_ranks=ranks,sp_group_world_size=world,sp_size=size,sp_rank=rank,
            local_tile_indices=[int(v) for v in local],local_tile_count=len(local),
            max_tiles_per_rank=-(-facts['num_tiles']//size))
    facts['stack_tiling']=bool(getattr(model,'stack_tiling',False))
    for name in ('clip_length','tokens_chunk_size','token_overlap','vae_ratio_t','token_drop','isolated_first_frame','isolated_key_frame'):
        value=getattr(model,name,None)
        facts[name]=value if isinstance(value,(int,bool,type(None))) else str(value)
    return facts


def diagnostic_rpc(worker):
    import torch
    import torch_npu
    import numpy as np
    from vllm.config import set_current_vllm_config
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size
    rank=worker.rank;torch.npu.set_device(worker.device)
    pipeline=worker.model_runner.pipeline;video_vae=pipeline.video_vae
    path=ROOT/f'vae_rank{rank}138.json';assert not path.exists()
    report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'repeats':[],
        'sampler_evaluations':0,'DiT_forward_calls':0,'no_math_change':True,'no_candidate':True,
        'initial_runtime':runtime(torch),'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    from cohort128 import Cohort128,publish_fatal128
    cohort=Cohort128(torch,prefix='R138videoVAE',timeout_s=900)
    try:
        state=report['initial_runtime']
        assert not state['deterministic'] and not state['warn_only']
        assert not state['HF32_matmul'] and not state['HF32_conv'] and not state['parametrization_cache_enabled']
        assert get_tensor_model_parallel_world_size()==2 and get_sequence_parallel_world_size()==4
        report['TP_actual']=2;report['TP_group']=list(get_tp_group().ranks)
        report['SP_actual']=4;report['SP_group']=list(get_sp_group().ranks)
        # Original FP32 VAE parameters must be untouched by any installer.
        params=[(n,p) for n,p in video_vae.named_parameters()]
        assert params and all(p.dtype==torch.float32 for n,p in params if p.is_floating_point())
        report['vae_parameter_dtype']='torch.float32'
        report['vae_parameter_count']=len(params)
        report['vae_parameter_words']=sum(p.numel() for n,p in params)
        files=list(PRIOR.glob('final_latents_pid*127.pt'));assert len(files)==1
        source_sha=sha(files[0]);assert source_sha==preflight_latent_sha()
        bundle=torch.load(files[0],map_location='cpu',weights_only=True)
        assert bundle['actual_DiT_evaluations']==49
        latent_cpu=bundle['video_latent'];assert latent_cpu.dtype==torch.float32 and tuple(latent_cpu.shape)==(1,24,37,48,84)
        latent=latent_cpu.to(worker.device).clone()
        report['fixed_latent']={'source_file':str(files[0]),'source_file_SHA':source_sha,
            'shape':list(latent_cpu.shape),'dtype':str(latent_cpu.dtype),'words':latent_cpu.numel(),
            'cpu_vs_npu_complete_word_pairs':word_compare(latent_cpu,latent,torch,np),
            'input_fingerprint':fingerprint(latent,torch,include_pointers=True)}
        report['tiling']=tiling_facts(video_vae,latent,torch)
        report['state_before']=cohort.run('vae_state_before',lambda:component_state(video_vae,torch))
        first=None
        original_decode_base=video_vae.model.decode_base
        timings={}
        def timed_decode_base(*args,**kwargs):
            start=time.perf_counter()
            try:return original_decode_base(*args,**kwargs)
            finally:timings['decode_base_s']=time.perf_counter()-start
        def install_timer():
            video_vae.model.decode_base=timed_decode_base
        def restore_timer():
            if vars(video_vae.model).get('decode_base') is timed_decode_base:video_vae.model.decode_base=original_decode_base
        cohort.run('install_decode_base_timer',install_timer)
        try:
            for repeat in range(REPEATS):
                def call():
                    timings.clear()
                    before=fingerprint(latent,torch,include_pointers=True);rng_before=rng_state(torch)
                    load_start=time.perf_counter()
                    with pipeline._component_on_device(video_vae):
                        load_s=time.perf_counter()-load_start
                        torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
                        start=time.perf_counter()
                        decoded=video_vae.decode_latent(latent)
                        torch.npu.synchronize()
                        total_s=time.perf_counter()-start
                        shape=list(decoded.shape);dtype=str(decoded.dtype)
                        cpu=decoded.detach().cpu().contiguous().clone()
                    assert bool(torch.isfinite(cpu).all())
                    assert normalized_fingerprint(before)==normalized_fingerprint(fingerprint(latent,torch,include_pointers=True))
                    state_after=runtime(torch)
                    assert not state_after['deterministic'] and not state_after['warn_only'] and not state_after['HF32_matmul'] and not state_after['HF32_conv'] and not state_after['parametrization_cache_enabled']
                    return cpu,{'repeat':repeat,'load_s':load_s,'decode_latent_s':total_s,
                        'decode_base_s':timings.get('decode_base_s'),'residual_revert_normalize_copy_s':(total_s-timings.get('decode_base_s',0.0)),
                        'shape':shape,'dtype':dtype,'output_SHA':fingerprint(cpu,torch),
                        'runtime':state_after,'rng_before':rng_before,'rng_after':rng_state(torch),
                        'peak_allocated_GiB':torch.npu.max_memory_allocated()/(1024**3)}
                cpu,row=cohort.run(f'video_vae_native_repeat_{repeat}',call)
                if first is None:first=cpu.clone()
                row['output_vs_first']=word_compare(cpu,first,torch,np)
                report['repeats'].append(row)
                cohort.run(f'video_vae_repeat_{repeat}_publish',publish)
                del cpu
        finally:
            cohort.run('remove_decode_base_timer',restore_timer)
        assert vars(video_vae.model).get('decode_base',original_decode_base) is original_decode_base
        report['decode_base_timer_removed']=True
        report['state_after']=cohort.run('vae_state_after',lambda:component_state(video_vae,torch))
        assert same_original_state(report['state_before'],report['state_after']);report['original_parameter_state_unchanged']=True
        report['final_runtime']=runtime(torch);assert report['final_runtime']==report['initial_runtime']
        report['timer_restored_and_no_math_change']=True;report['completed']=True
        cohort.run('all8_video_vae_measurement_complete',publish)
        return {'rank':rank,'completed':True,'report':str(path),'repeats':REPEATS,'no_math_change':True}
    except BaseException as primary:
        report['failure']={'type':type(primary).__name__,'message':str(primary)};report['completed']=False
        try:publish_fatal128({'ok':False,'rank':rank,'pid':os.getpid(),'phase':'video_vae_RPC',**report['failure']})
        except BaseException:pass
        try:publish()
        except BaseException:pass
        raise


def preflight_latent_sha():
    preflight=json.loads((ROOT/'preflight138.json').read_text())
    assert preflight['completed'] and preflight['fixed_video_latent_SHA_verified']
    return preflight['fixed_video_latent_sha256']


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r138_native_video_vae_timing_repeat';assert not hasattr(DiffusionWorker,name)
    def method(self):
        try:return diagnostic_rpc(self)
        except BaseException as primary:
            try:publish_fatal128({'ok':False,'rank':self.rank,'pid':os.getpid(),'phase':'video_vae_RPC_entry','type':type(primary).__name__,'message':str(primary)})
            except BaseException:pass
            raise
    setattr(DiffusionWorker,name,method)