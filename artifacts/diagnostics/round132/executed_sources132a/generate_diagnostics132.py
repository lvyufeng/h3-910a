#!/usr/bin/env python3
"""Formal8worker audio first-change RPC only; no generation or baseline adoption."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_audio13201')
SESSION=ROOT/'engine_audio132a'


def main():
    receipt=SESSION/'diagnostic_lifecycle132.json';assert SESSION.is_dir() and not receipt.exists()
    approval=json.loads((ROOT/'approval132.json').read_text());assert not approval['adopt_deterministic_baseline'] and not approval['copy_candidate']
    os.environ['H3_R127_SESSION_OUT']=str(SESSION)
    from full_bootstrap127 import install_alignment
    install_alignment()
    from vllm_omni.entrypoints.omni import Omni
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'kwargs':kwargs,'approval':approval,'RPCs':[],
        'no_generate_or_sampler':True,'copy_candidate_enabled':False,'new_numeric_baseline_adopted':False}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    try:
        started=time.perf_counter();omni=Omni(model='/tmp/h3_align12401/original_metadata124f/FL2VA',**kwargs)
        report['engine_initialization_s']=time.perf_counter()-started;publish()
        started=time.perf_counter();reply=omni.engine.collective_rpc('r132_audio_first_change_diagnostic',timeout=2400,args=(),stage_ids=[0])
        ranks=[json.loads(p.read_text()) for p in ROOT.glob('audio_tail_rank*132.json')]
        assert len(ranks)==8 and sorted(r['rank'] for r in ranks)==list(range(8)) and len({r['pid'] for r in ranks})==8
        assert all(r['completed'] and len(r['passes'])==9 and r['private_flags_physically_restored'] for r in ranks)
        checks=sum(sum(p['strict_Half_checks'] for p in r['passes']) for r in ranks);assert checks==11160
        first=[{'rank':r['rank'],'boundary':r['tail_trace_compare']['first_different_boundary']} for r in ranks]
        local_calls=sum(r['fixed_full_input_audio_local_replay']['actual_local_method_calls'] for r in ranks)
        assert local_calls==80 and all(len(r['fixed_full_input_audio_local_replay']['rows'])==10 for r in ranks)
        local=[{'rank':r['rank'],'repeat_changed':any(not row['full_output_vs_replay_first']['all_words_equal'] for row in r['fixed_full_input_audio_local_replay']['rows'])} for r in ranks]
        report['RPCs'].append({'mode':'audio_tail_and_original_local_repeats','host_wall_s':time.perf_counter()-started,'reply':reply,
            'executed8rank_receipts':True,'same8worker_pids':sorted(r['pid'] for r in ranks),'original_strict_checks_allranks':checks,
            'actual_local_audio_method_calls':80,'first_trace_boundaries':first,'fixed_local_repeats':local})
        report['completed']=True;publish();print(json.dumps({'completed':True,'checks':checks,'first':first,'local':local,'flags_restored':True}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
