#!/usr/bin/env python3
"""Launch pinned no-sampling formal-worker diagnostics with bounded private control."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import select
import time

ROOT=Path('/tmp/h3_full5013501')
SESSION=ROOT/'engine_candidate135a'


def main():
    receipt=ROOT/'engine_candidate135a_process.json';assert not SESSION.exists() and not receipt.exists(),'Refuse replay of uncertain launch'
    preflight=json.loads((ROOT/'preflight127.json').read_text());assert preflight['completed'] and not preflight['weights_loaded']
    input_probe=json.loads((ROOT/'input_preflight129.json').read_text());assert input_probe['completed']
    fresh=json.loads((ROOT/'preflight135.json').read_text());assert fresh['completed']
    assert sorted(row['rank'] for row in fresh['all8_input_fixture_SHA_reverified'])==list(range(8))
    pins={'/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
        '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
        '/tmp/h3_align12301/same_initial_sampling123.pt':'8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9'}
    for path,digest in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for name,entry in preflight['sources'].items():assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert not (ROOT/'cohort_failure128.json').exists()
    assert not list(ROOT.glob('full50_rank*135.json'))
    approval=json.loads((ROOT/'approval135.json').read_text());assert not approval['baseline_adoption'] and approval['preserve_original50_TP2_USP4_155checks_FP32_exceptions']
    from cohort128 import prepare_failure_watch128,monitor_private_failure
    failure_watch=prepare_failure_watch128()
    SESSION.mkdir()
    key_path=ROOT/'private_IPC_key127.bin';assert not key_path.exists()
    fd=os.open(key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(os.urandom(32))
    work=ROOT/'cann_engine_candidate135a';work.mkdir(exist_ok=False)
    assignments={'H3_R127_IPC_KEY_FILE':str(key_path),'H3_R127_SELECTIVE_BOOTSTRAP':'1','H3_R127_SESSION_OUT':str(SESSION),
        'H3_R127_INIT_DIAGNOSTICS':'1',
        'H3_NPU_ALIGNMENT124_BOOTSTRAP':'0','H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP':'0',
        'H3_R126_FULL_SELECTIVE_BOOTSTRAP':'0','H3_R126_VAE_OPERAND_CACHE':'0',
        'H3_NPU_NAN_COMPAT124':'1','H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1',
        'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200','VLLM_OMNI_910A_FLASH_ATTN':'1',
        'VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16','VLLM_OMNI_910A_TEXT_HF_MATH':'1',
        'VLLM_OMNI_910A_NO_FUSED_ACT':'1','VLLM_OMNI_910A_NO_FUSED_NORM':'1',
        'VLLM_OMNI_910A_SHARED_STREAM':'1','VLLM_OMNI_910A_FAST_RESIDUAL':'0','VLLM_OMNI_910A_FAST_ACT':'0',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
        'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','ASCEND_WORK_PATH':str(work)}
    assert all(' ' not in v and '\n' not in v for v in assignments.values())
    shell='source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_full5013501:/tmp/h3_align12401:$PYTHONPATH; export '
    shell+=' '.join(k+'='+v for k,v in assignments.items())
    shell+='; exec /home/mseco/vllm910a/bin/python -u /tmp/h3_full5013501/generate_diagnostics135.py'
    command=['/bin/bash','-c',shell]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    deadline_monotonic=time.monotonic()+5400
    with (ROOT/'engine_candidate135a.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row={'pid':process.pid,'launcher_pid':os.getpid(),'command':command,'source_sha256':sources,'original_source_sha256':pins,
         'start_unix':time.time(),'compute_budget_seconds':5400,'cleanup_grace_seconds':360,'new_precision_policy_accepted':False,'model_weights_transferred':False}
    def publish():
        if receipt.exists():assert json.loads(receipt.read_text())['pid']==process.pid
        temporary=ROOT/'engine_candidate135a_process.new'
        assert temporary.resolve()==Path('/tmp/h3_full5013501/engine_candidate135a_process.new')
        assert receipt.resolve()==Path('/tmp/h3_full5013501/engine_candidate135a_process.json')
        temporary.write_text(json.dumps(row,indent=2)+'\n');os.replace(temporary,receipt)
    publish();print(json.dumps({'started':str(SESSION),'pid':process.pid}),flush=True)
    stop=threading.Event();monitor_errors=[]
    def watch():
        try:monitor_private_failure(process,stop,failure_watch,deadline_monotonic=deadline_monotonic)
        except BaseException as exc:
            monitor_errors.append({'type':type(exc).__name__,'message':str(exc)})
            if not getattr(process,'_private_abort_called135',False):
                from private_abort135 import private_rows135,abort_private135
                if private_rows135():
                    try:abort_private135()
                    except BaseException as cleanup:
                        monitor_errors.append({'type':type(cleanup).__name__,'message':str(cleanup)})
    watcher=threading.Thread(target=watch,name='R131PrivateCohortFailure',daemon=True);watcher.start()
    parent_fd=os.pidfd_open(process.pid)
    try:ready,_,_=select.select([parent_fd],[],[],max(0,deadline_monotonic-time.monotonic())+360)
    finally:os.close(parent_fd)
    if not ready:
        stop.set();watcher.join(timeout=1)
        row['exit_code']=None;row['completion_unknown']=True;row['cohort_monitor_errors']=monitor_errors
        row['cleanup_failure']={'type':'TimeoutError','message':'Parent teardown exceeded compute deadline plus cleanup grace; retain lease and unknown state'}
        publish();raise TimeoutError(row['cleanup_failure']['message'])
    rc=process.wait();stop.set();watcher.join(timeout=360)
    row['exit_code']=rc;row['end_unix']=time.time();row['cohort_monitor_errors']=monitor_errors
    if watcher.is_alive():
        row['cleanup_failure']={'type':'TimeoutError','message':'Private monitor did not complete bounded cleanup'}
        publish();raise TimeoutError(row['cleanup_failure']['message'])
    from private_abort135 import private_rows135,abort_private135
    try:
        remaining=private_rows135();row['private_children_after_parent_exit']=remaining
        if remaining:row['post_parent_exit_private_cleanup']=abort_private135()
        assert not private_rows135(),'Private children remain after parent exit'
        row['all_private_children_exited']=True
    except BaseException as exc:
        row['cleanup_failure']={'type':type(exc).__name__,'message':str(exc)};publish();raise
    publish()
    print(json.dumps({'completed_process':process.pid,'exit_code':rc}),flush=True);raise SystemExit(rc)


if __name__=='__main__':main()
