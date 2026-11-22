#!/usr/bin/env python3
"""One original8worker component diagnostic RPC, never sampling or candidate math."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_components13703');SESSION=ROOT/'engine_components137c'


def main():
    assert SESSION.is_dir();receipt=SESSION/'component_lifecycle137.json';assert not receipt.exists()
    approval=json.loads((ROOT/'approval137.json').read_text());assert approval['original_math_and_parameters_unchanged'] and not approval['full49_generation']
    os.environ['H3_R127_SESSION_OUT']=str(SESSION)
    from full_bootstrap127 import install_alignment
    install_alignment()
    from vllm_omni.entrypoints.omni import Omni
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'approval':approval,'kwargs':kwargs,'sampler_evaluations':0,'DiT_forward_calls':0,
        'candidate_enabled':False,'numeric_tolerance_adopted':False,'RPCs':[]}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    try:
        started=time.perf_counter();omni=Omni(model='/tmp/h3_align12401/original_metadata124f/FL2VA',**kwargs)
        report['engine_initialization_s']=time.perf_counter()-started;publish()
        started=time.perf_counter();reply=omni.engine.collective_rpc('r137_original_components_fixed_repeat',timeout=2400,args=(),stage_ids=[0])
        rows=[json.loads(p.read_text()) for p in ROOT.glob('component_rank*137.json')]
        assert len(rows)==8 and sorted(r['rank'] for r in rows)==list(range(8)) and len({r['pid'] for r in rows})==8
        assert all(r['completed'] and r['original_encoder_method_restored'] and r['final_runtime']==r['initial_runtime'] for r in rows)
        for r in rows:
            assert [c['component'] for c in r['components']]==['encoder','audio_decoder']
            assert all(len(c['passes'])==14 and c['original_parameters_buffers_and_CPU_masters_unchanged'] and c['own_observers_removed'] for c in r['components'])
            assert r['sampler_evaluations']==r['DiT_forward_calls']==0
        report['RPCs'].append({'host_wall_s':time.perf_counter()-started,'reply':reply,'actual8_worker_receipts':True,'worker_pids':sorted(r['pid'] for r in rows),
            'encoder_fixed_repeats_allranks':112,'encoder_preparation_calls_allranks':8,'audio_fixed_repeats_allranks':112,
            'actual_component_phases':'native8 / trace2 / observer_restored4 per rank',
            'native_default_false_preserved':True,'no_sampler_or_replacement_math':True})
        report['business_work_completed']=True;publish()
    except BaseException as exc:
        report['failure']={'type':type(exc).__name__,'message':str(exc)};report['completed']=False
        try:publish()
        except BaseException:pass
        raise
    finally:
        import sys
        primary=sys.exc_info()[1];close_error=None
        if omni is not None:
            try:omni.close();report['engine_close_completed']=True
            except BaseException as exc:
                close_error=exc;report['engine_close_completed']=False;report['completed']=False
                report['cleanup_failure']={'type':type(exc).__name__,'message':str(exc)}
        if primary is None and close_error is None and report.get('business_work_completed'):
            report['completed']=True
        try:publish()
        except BaseException:
            if primary is None and close_error is None:raise
        if close_error is not None and primary is None:raise close_error
    print(json.dumps({'completed':True,'worker_count':8,'fixed_repeats_per_component':112,'no_math_change':True}),flush=True)


if __name__=='__main__':main()
