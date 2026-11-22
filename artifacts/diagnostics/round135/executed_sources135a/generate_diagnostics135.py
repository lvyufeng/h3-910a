#!/usr/bin/env python3
"""One formal8worker full50 candidate diagnostic only, never generation/adoption."""
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_full5013501')
SESSION=ROOT/'engine_candidate135a'


def main():
    receipt=SESSION/'diagnostic_lifecycle135.json';assert SESSION.is_dir() and not receipt.exists()
    approval=json.loads((ROOT/'approval135.json').read_text());assert approval['selected']=='只做全50层候选诊断 (Recommended)' and not approval['baseline_adoption']
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
        started=time.perf_counter();reply=omni.engine.collective_rpc('r135_full50_final_AdaLN_candidate_diagnostic',timeout=2400,args=(),stage_ids=[0])
        ranks=[json.loads(p.read_text()) for p in ROOT.glob('full50_rank*135.json')]
        assert len(ranks)==8 and sorted(r['rank'] for r in ranks)==list(range(8)) and len({r['pid'] for r in ranks})==8
        assert all(r['completed'] and len(r['passes'])==9 and r['candidate_method_physically_restored'] and r['private_flags_physically_restored'] for r in ranks)
        checks=sum(sum(p['strict_Half_checks'] for p in r['passes']) for r in ranks);assert checks==11160
        assert sum(r['candidate_actual_local_calls'] for r in ranks)==24
        cases=[]
        for case in ('original','candidate','restored'):
            rows=[p for r in ranks for p in r['passes'] if p['case']==case];assert len(rows)==24
            assert all(p['actual_deterministic_flag'] for p in rows)
            cases.append({'case':case,'all8_both_predictions_repeat_bit_identical':all(all(v['all_words_equal'] for v in p['output_vs_same_case_first']) for p in rows)})
        report['RPCs'].append({'mode':'full50_final_AdaLN_candidate','host_wall_s':time.perf_counter()-started,'reply':reply,
            'executed8rank_receipts':True,'same8worker_pids':sorted(r['pid'] for r in ranks),'strict_checks_allranks':checks,
            'actual_candidate_local_calls':24,'cases':cases,'candidate_method_and_flags_restored':True})
        report['completed']=True;publish();print(json.dumps({'completed':True,'checks':checks,'cases':cases,'no_adoption':True}),flush=True)
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
