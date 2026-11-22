#!/usr/bin/env python3
"""One formal eight-worker full50 diagnostic RPC, never generate or sampler steps."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_full5013101')
SESSION=ROOT/'engine_deterministic131a'


def main():
    assert SESSION.is_dir();receipt=SESSION/'diagnostic_lifecycle131.json';assert not receipt.exists()
    approval=json.loads((ROOT/'approval131.json').read_text())
    assert approval['selected']=='只做全50层诊断 (Recommended)' and not approval['model_candidate_adoption_authorized']
    os.environ['H3_R127_SESSION_OUT']=str(SESSION)
    from full_bootstrap127 import install_alignment
    install_alignment()
    from vllm_omni.entrypoints.omni import Omni
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'kwargs':kwargs,'approval':approval,'no_sampling_or_generate':True,
        'copy_candidate_enabled':False,'new_precision_policy_adopted':False,'RPCs':[]}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    try:
        started=time.perf_counter();omni=Omni(model='/tmp/h3_align12401/original_metadata124f/FL2VA',**kwargs)
        report['engine_initialization_s']=time.perf_counter()-started;publish()
        started=time.perf_counter()
        reply=omni.engine.collective_rpc('r131_full50_deterministic_diagnostic',timeout=2400,args=(),stage_ids=[0])
        reports=[json.loads(p.read_text()) for p in ROOT.glob('full50_rank*131.json')]
        assert len(reports)==8 and sorted(r['rank'] for r in reports)==list(range(8))
        assert all(r['completed'] and len(r['passes'])==9 and r['private_flags_physically_restored'] for r in reports)
        assert len({r['pid'] for r in reports})==8
        checks=sum(sum(p['strict_Half_checks'] for p in r['passes']) for r in reports);assert checks==11160
        cases=[]
        for case in ('native','deterministic','restored'):
            rows=[p for r in reports for p in r['passes'] if p['case']==case]
            assert len(rows)==24 and all(p['actual_deterministic_flag']==(case=='deterministic') for p in rows)
            cases.append({'case':case,'all8_repeats_bit_identical':all(all(v['all_words_equal'] for v in p['output_vs_same_case_first']) for p in rows),
                'all8_time_repeats_bit_identical':all(p['time_vs_same_case_first']['all_words_equal'] for p in rows)})
        report['RPCs'].append({'mode':'full50_native_deterministic_restored','host_wall_s':time.perf_counter()-started,'reply':reply,
            'actual8rank_execution':True,'same_worker_pids':sorted(r['pid'] for r in reports),'forwards_each_rank':9,
            'strict_checks_allranks':checks,'cases':cases,'all_flags_restored':True})
        report['completed']=True;publish()
        print(json.dumps({'completed':True,'checks':checks,'cases':cases,'copy_candidate_enabled':False,'flags_restored':True}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
