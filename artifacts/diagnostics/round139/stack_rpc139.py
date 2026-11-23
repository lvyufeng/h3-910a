"""Evaluate the native `stack_tiling` microbatch candidate on the video VAE.

Phases in one process on the same fixed latent: native (4 repeats) -> candidate
(native stack_tiling=True, 4 repeats) -> restored (native again, 4 repeats).
The pure-native restored phase is a control: it is judged by the same user
approved rule, so an unsatisfiable rule is reported instead of being relaxed.

User rule (Issue #7, option 1): a candidate passes only if its deviation from
the original is no worse than the native repeat-to-repeat deviation, in both
differing uint8 component fraction and maxabs. No other math is installed.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from fingerprints129 import fingerprint, normalized_fingerprint
from original_state137 import component_state, same_original_state

ROOT=Path('/tmp/h3_vae13902')
PRIOR=Path('/tmp/h3_quality13601/engine_quality136a/r01_original_false')
REPEATS=4
RULE_PATH='acceptance_rule139.json'


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


def diff(a,b,torch,np):
    """Complete-domain uint8 word comparison of two decoded videos."""
    a=a.reshape(-1);b=b.reshape(-1)
    assert a.numel()==b.numel() and a.dtype==b.dtype
    av=a.numpy();bv=b.numpy()
    changed=int(np.count_nonzero(av!=bv));total=int(av.size)
    if changed:
        delta=np.abs(av.astype(np.int16)-bv.astype(np.int16))
        maxabs=int(delta.max())
        rms=float(np.sqrt(np.mean(delta.astype(np.float64)**2)))
        ref=float(np.sqrt(np.mean(bv.astype(np.float64)**2)))
        return {'different_words':changed,'total_words':total,
            'fraction_percent':100.0*changed/total,'max_abs_LSB':maxabs,
            'relative_RMS_percent':(100.0*rms/ref) if ref else None}
    return {'different_words':0,'total_words':total,'fraction_percent':0.0,'max_abs_LSB':0,'relative_RMS_percent':0.0}


def band(pairs,label):
    return {'label':label,'pairs':len(pairs),'max_fraction_percent':max(p['fraction_percent'] for p in pairs),
        'max_abs_LSB':max(p['max_abs_LSB'] for p in pairs),
        'max_relative_RMS_percent':max(p['relative_RMS_percent'] or 0.0 for p in pairs),
        'any_identical':any(p['different_words']==0 for p in pairs),
        'all':pairs}


def diagnostic_rpc(worker):
    import torch
    import torch_npu
    import numpy as np
    from vllm_omni.diffusion.distributed.parallel_state import get_sp_group,get_sequence_parallel_world_size
    from vllm.distributed.parallel_state import get_tp_group,get_tensor_model_parallel_world_size
    rank=worker.rank;torch.npu.set_device(worker.device)
    pipeline=worker.model_runner.pipeline;video_vae=pipeline.video_vae;model=video_vae.model
    path=ROOT/f'stack_rank{rank}139.json';assert not path.exists()
    rule=json.loads((ROOT/RULE_PATH).read_text())
    report={'scope':__doc__,'rank':rank,'pid':os.getpid(),'completed':False,'rule':rule,
        'phases':{},'sampler_evaluations':0,'DiT_forward_calls':0,'candidate_name':'native stack_tiling microbatch',
        'initial_runtime':runtime(torch),'source_sha256':{p.name:sha(p) for p in ROOT.glob('*.py')}}
    def publish():path.write_text(json.dumps(report,indent=2)+'\n')
    from cohort128 import Cohort128,publish_fatal128
    cohort=Cohort128(torch,prefix='R139stackTiling',timeout_s=1800)
    try:
        state=report['initial_runtime']
        assert not state['deterministic'] and not state['warn_only']
        assert not state['HF32_matmul'] and not state['HF32_conv'] and not state['parametrization_cache_enabled']
        assert get_tensor_model_parallel_world_size()==2 and get_sequence_parallel_world_size()==4
        report['TP_actual']=2;report['SP_actual']=4
        assert all(p.dtype==torch.float32 for n,p in video_vae.named_parameters() if p.is_floating_point())
        preflight=json.loads((ROOT/'preflight139.json').read_text());assert preflight['completed']
        files=list(PRIOR.glob('final_latents_pid*127.pt'));assert len(files)==1
        source_sha=sha(files[0]);assert source_sha==preflight['fixed_video_latent_sha256']
        bundle=torch.load(files[0],map_location='cpu',weights_only=True);assert bundle['actual_DiT_evaluations']==49
        latent_cpu=bundle['video_latent'];assert latent_cpu.dtype==torch.float32 and tuple(latent_cpu.shape)==(1,24,37,48,84)
        latent=latent_cpu.to(worker.device).clone()
        report['fixed_latent']={'source_file':str(files[0]),'source_file_SHA':source_sha,'shape':list(latent_cpu.shape),
            'dtype':str(latent_cpu.dtype),'fingerprint':fingerprint(latent,torch,include_pointers=True)}
        original_stack_tiling=bool(getattr(model,'stack_tiling',False))
        report['candidate_knob']={'name':'stack_tiling','original_value':original_stack_tiling,'candidate_value':not original_stack_tiling,
            'native_config_option':True,'tile_windows_or_blend_changed':False,'parameters_changed':False}
        num_tiles=int(video_vae._decoder_tile_count(latent))
        sp_size=int(video_vae.parallel_size)
        report['geometry']={'num_tiles':num_tiles,'parallel_size':sp_size,'tiles_per_rank':-(-num_tiles//sp_size),
            'tile_rows':int(model.tile_rows) if hasattr(model,'tile_rows') else None}
        report['state_before']=cohort.run('vae_state_before',lambda:component_state(video_vae,torch))
        outputs={};timing={}
        def run_phase(phase,stack_tiling,repeats):
            def set_knob():model.stack_tiling=bool(stack_tiling)
            cohort.run(f'{phase}_set_stack_tiling_{bool(stack_tiling)}',set_knob)
            rows=[]
            for repeat in range(repeats):
                def call():
                    before=fingerprint(latent,torch,include_pointers=True)
                    start=time.perf_counter()
                    with pipeline._component_on_device(video_vae):
                        torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
                        begin=time.perf_counter()
                        decoded=video_vae.decode_latent(latent)
                        torch.npu.synchronize()
                        elapsed=time.perf_counter()-begin
                        peak=torch.npu.max_memory_allocated()/(1024**3)
                        cpu=decoded.detach().cpu().contiguous().clone()
                    assert bool(torch.isfinite(cpu.float()).all())
                    assert normalized_fingerprint(before)==normalized_fingerprint(fingerprint(latent,torch,include_pointers=True))
                    runtime_after=runtime(torch)
                    assert runtime_after==state
                    return cpu,{'phase':phase,'repeat':repeat,'stack_tiling':bool(stack_tiling),
                        'decode_s':elapsed,'peak_allocated_GiB':peak,'shape':list(cpu.shape),'dtype':str(cpu.dtype),
                        'output_SHA':fingerprint(cpu,torch),'runtime':runtime_after}
                cpu,row=cohort.run(f'{phase}_repeat_{repeat}',call)
                rows.append(row);outputs.setdefault(phase,[]).append(cpu)
                cohort.run(f'{phase}_repeat_{repeat}_publish',publish)
            timing[phase]=rows
            return rows
        run_phase('native',False,REPEATS)
        run_phase('candidate',True,REPEATS)
        cohort.run('restore_stack_tiling',lambda:setattr(model,'stack_tiling',original_stack_tiling))
        run_phase('restored',original_stack_tiling,REPEATS)
        assert bool(model.stack_tiling)==original_stack_tiling
        report['knob_restored']=True
        # Pairwise bands on the complete uint8 domain.
        native=outputs['native'];candidate=outputs['candidate'];restored=outputs['restored']
        npairs=[]
        for i in range(len(native)):
            for j in range(i+1,len(native)):npairs.append(diff(native[i],native[j],torch,np))
        cpairs=[diff(c,n,None,np) for c in candidate for n in native]
        rpairs=[diff(r,n,None,np) for r in restored for n in native]
        crpairs=[diff(r,c,None,np) for r in restored for c in candidate]
        rnpairs=[]
        for i in range(len(restored)):
            for j in range(i+1,len(restored)):rnpairs.append(diff(restored[i],restored[j],torch,np))
        native_band=band(npairs,'native_vs_native')
        candidate_band=band(cpairs,'candidate_vs_native')
        restored_band=band(rpairs,'restored_vs_native_control')
        restored_internal=band(rnpairs,'restored_vs_restored_control_internal')
        candidate_internal=band([diff(candidate[i],candidate[j],torch,np) for i in range(len(candidate)) for j in range(i+1,len(candidate))],'candidate_vs_candidate')
        candidate_vs_restored=band(crpairs,'candidate_vs_restored')
        verdict={'rule':rule['rule'],
            'native_band':{k:native_band[k] for k in ('max_fraction_percent','max_abs_LSB','any_identical')},
            'candidate_vs_native':{k:candidate_band[k] for k in ('max_fraction_percent','max_abs_LSB','any_identical')},
            'restored_control_vs_native':{k:restored_band[k] for k in ('max_fraction_percent','max_abs_LSB','any_identical')},
            'candidate_within_band':bool(candidate_band['max_fraction_percent']<=native_band['max_fraction_percent'] and candidate_band['max_abs_LSB']<=native_band['max_abs_LSB']),
            'restored_control_within_band':bool(restored_band['max_fraction_percent']<=native_band['max_fraction_percent'] and restored_band['max_abs_LSB']<=native_band['max_abs_LSB'])}
        verdict['rule_satisfiable_by_pure_native_control']=verdict['restored_control_within_band']
        verdict['candidate_passes']=bool(verdict['candidate_within_band'] and verdict['rule_satisfiable_by_pure_native_control'])
        report['bands']={'native_vs_native':native_band,'candidate_vs_native':candidate_band,
            'restored_vs_native_control':restored_band,'restored_vs_restored_control_internal':restored_internal,
            'candidate_vs_candidate':candidate_internal,'candidate_vs_restored':candidate_vs_restored}
        report['verdict']=verdict
        for phase,rows in timing.items():
            report['phases'][phase]=[{'repeat':r['repeat'],'decode_s':r['decode_s'],
                'peak_allocated_GiB':r['peak_allocated_GiB'],'output_SHA':r['output_SHA']['sha256_logical_words'],
                'stack_tiling':r['stack_tiling']} for r in rows]
        report['state_after']=cohort.run('vae_state_after',lambda:component_state(video_vae,torch))
        assert same_original_state(report['state_before'],report['state_after']);report['original_parameter_state_unchanged']=True
        report['final_runtime']=runtime(torch);assert report['final_runtime']==report['initial_runtime']
        report['completed']=True
        cohort.run('all8_stack_candidate_complete',publish)
        return {'rank':rank,'completed':True,'report':str(path),'candidate_passes':verdict['candidate_passes']}
    except BaseException as primary:
        report['failure']={'type':type(primary).__name__,'message':str(primary)};report['completed']=False
        try:model.stack_tiling=locals().get('original_stack_tiling',model.stack_tiling)
        except BaseException:pass
        try:publish_fatal128({'ok':False,'rank':rank,'pid':os.getpid(),'phase':'stack_candidate_RPC',**report['failure']})
        except BaseException:pass
        try:publish()
        except BaseException:pass
        raise


def install_rpc():
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
    name='r139_video_vae_stack_tiling_candidate';assert not hasattr(DiffusionWorker,name)
    def method(self):
        try:return diagnostic_rpc(self)
        except BaseException as primary:
            try:publish_fatal128({'ok':False,'rank':self.rank,'pid':os.getpid(),'phase':'stack_RPC_entry','type':type(primary).__name__,'message':str(primary)})
            except BaseException:pass
            raise
    setattr(DiffusionWorker,name,method)