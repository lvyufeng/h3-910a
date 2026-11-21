#!/usr/bin/env python3
"""Initialize one formal8-worker engine and send read-only repeat/trace RPCs.

No generate call, sampler step, candidate stream/cache or weights download.
Diagnostic forwards are original50 blocks/155checks on native worker threads.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time

ROOT=Path('/tmp/h3_repeat12901')
SESSION=ROOT/'engine_repeat129a'


def main():
    assert SESSION.is_dir();receipt=SESSION/'diagnostic_lifecycle129.json';assert not receipt.exists()
    os.environ['H3_R127_SESSION_OUT']=str(SESSION)
    from full_bootstrap127 import install_alignment
    install_alignment()
    from vllm_omni.entrypoints.omni import Omni
    kwargs={'trust_remote_code':True,'task_type':'fl2va','num_gpus':8,'tensor_parallel_size':2,'text_encoder_tp_size':8,
        'ulysses_degree':4,'ring_degree':1,'dtype':'half','enforce_eager':True,'enable_layerwise_offload':True,
        'vae_use_tiling':True,'vae_parallel_mode':'tile','vae_patch_parallel_size':8,
        'diffusion_attention_backend':'TORCH_SDPA','stage_init_timeout':2400,'init_timeout':2400}
    report={'scope':__doc__,'completed':False,'kwargs':kwargs,'RPCs':[],'no_sampling_or_generate':True,
        'no_copy_stream_candidate_enabled':True,'own_live_prior_NPU_fixture_used':True}
    def publish():receipt.write_text(json.dumps(report,indent=2)+'\n')
    publish();omni=None
    try:
        start=time.perf_counter();omni=Omni(model='/tmp/h3_align12401/original_metadata124f/FL2VA',**kwargs)
        report['engine_initialization_s']=time.perf_counter()-start;publish()
        for mode in ('repeat','trace','detailed'):
            start=time.perf_counter()
            reply=omni.engine.collective_rpc('r129_native_repeatability',timeout=1800,args=(mode,),stage_ids=[0])
            files=list(ROOT.glob(mode+'_rank*129.json'));assert len(files)==8
            results=[json.loads(p.read_text()) for p in files]
            assert sorted(r['rank'] for r in results)==list(range(8)) and all(r['completed'] for r in results)
            report['RPCs'].append({'mode':mode,'host_wall_s':time.perf_counter()-start,'reply':reply,
                'executed8rank_receipts':True,'forwards_each_rank':len(results[0]['passes']),
                'original_strict_checks_allranks':sum(sum(p['strict_Half_checks'] for p in r['passes']) for r in results),
                'first_trace_changes':[{'rank':r['rank'],'first':r['trace_compare'][0]['first_different_trace_boundary']} for r in results if r['trace_compare']]})
            publish();print(json.dumps({'RPC_completed':mode,'all8receipts':True,'first_changes':report['RPCs'][-1]['first_trace_changes']}),flush=True)
        report['completed']=True;report['actual_original_full50_forward_checks']=sum(r['original_strict_checks_allranks'] for r in report['RPCs']);publish()
    except BaseException as exc:report['failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    finally:
        if omni is not None:omni.close()


if __name__=='__main__':main()
