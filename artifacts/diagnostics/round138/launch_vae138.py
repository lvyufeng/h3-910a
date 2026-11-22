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

ROOT=Path('/tmp/h3_vae13803')
SESSION=ROOT/'engine_vae138c'


def guarded_launch137(command,child_env,log_path,row,publish,failure_watch,deadline_monotonic):
    """Own every spawned process even if publishing/thread/pidfd setup fails."""
    from private_abort137 import private_rows137,abort_private137
    from cohort128 import monitor_private_failure
    process=None;parent_fd=None;watcher=None;watcher_started=False
    stop=threading.Event();monitor_errors=[]
    def watch():
        try:monitor_private_failure(process,stop,failure_watch,deadline_monotonic=deadline_monotonic)
        except BaseException as exc:
            monitor_errors.append({'type':type(exc).__name__,'message':str(exc)})
            try:
                if private_rows137():abort_private137()
            except BaseException as cleanup:monitor_errors.append({'type':type(cleanup).__name__,'message':str(cleanup)})
    def best_effort_publish():
        try:publish()
        except BaseException as exc:row['receipt_publish_failure']={'type':type(exc).__name__,'message':str(exc)}
    try:
        with log_path.open('xb') as log:
            # The exact private capability exists at child exec, before shell
            # exports, so an early controller failure can still identify it.
            process=subprocess.Popen(command,cwd=ROOT,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row.update(pid=process.pid,start_unix=time.time());publish()
        print(json.dumps({'started':str(SESSION),'pid':process.pid}),flush=True)
        watcher=threading.Thread(target=watch,name='R138PrivateCohortFailure',daemon=True)
        watcher.start();watcher_started=True
        parent_fd=os.pidfd_open(process.pid)
        ready,_,_=select.select([parent_fd],[],[],max(0,deadline_monotonic-time.monotonic())+360)
        if not ready:raise TimeoutError('Parent exceeded compute deadline plus cleanup grace')
        rc=process.wait();stop.set();watcher.join(timeout=360)
        row.update(exit_code=rc,end_unix=time.time(),cohort_monitor_errors=monitor_errors)
        if watcher.is_alive():raise TimeoutError('Private monitor did not finish bounded cleanup')
        remaining=private_rows137();row['private_children_after_parent_exit']=remaining
        if remaining:row['post_parent_exit_private_cleanup']=abort_private137()
        assert not private_rows137(),'Private children remain after parent exit'
        row['all_private_children_exited']=True;row['completion_unknown']=False;publish()
        return rc
    except BaseException as primary:
        stop.set();row['completion_unknown']=True
        row['control_failure']={'type':type(primary).__name__,'message':str(primary)}
        try:
            if process is not None:
                if private_rows137():row['control_failure_private_cleanup']=abort_private137()
                assert not private_rows137(),'Private capability holders remain after controller failure'
                row['exit_code']=process.wait(timeout=360);row['all_private_children_exited']=True
            else:row['no_model_process_spawned']=True
        except BaseException as cleanup:
            row['cleanup_failure']={'type':type(cleanup).__name__,'message':str(cleanup)}
        row['cohort_monitor_errors']=monitor_errors;best_effort_publish()
        raise
    finally:
        import sys
        primary=sys.exc_info()[1];errors=[];stop.set()
        if parent_fd is not None:
            try:os.close(parent_fd)
            except BaseException as exc:errors.append({'operation':'close_parent_pidfd','message':str(exc)})
        if watcher_started or (watcher is not None and getattr(watcher,'ident',None) is not None):
            try:
                watcher.join(timeout=360)
                if watcher.is_alive():errors.append({'operation':'join_private_monitor','message':'Thread remains active'})
            except BaseException as exc:errors.append({'operation':'join_private_monitor','message':str(exc)})
        else:
            try:os.close(failure_watch)
            except BaseException as exc:errors.append({'operation':'close_unowned_failure_watch','message':str(exc)})
        if errors:
            row['control_cleanup_failures']=errors;row['completion_unknown']=True;best_effort_publish()
            if primary is None:raise RuntimeError('Controller cleanup incomplete: '+json.dumps(errors))


def main():
    receipt=ROOT/'engine_vae138c_process.json';assert not SESSION.exists() and not receipt.exists(),'Refuse replay of uncertain launch'
    preflight=json.loads((ROOT/'preflight127.json').read_text());assert preflight['completed'] and not preflight['weights_loaded']
    fresh=json.loads((ROOT/'preflight138.json').read_text());assert fresh['completed']
    assert fresh['fixed_video_latent_SHA_verified'] and fresh['native_source_SHA_verified']
    assert fresh['correct_kwarg_names_bind'] and fresh['wrong_kwarg_rejected']
    pins={'/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
        '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b'}
    for path,digest in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for name,entry in preflight['sources'].items():assert hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()==entry['sha256']
    assert not (ROOT/'cohort_failure128.json').exists()
    assert not list(ROOT.glob('vae_rank*138.json'))
    contract=json.loads((ROOT/'issue7_contract138.json').read_text())
    assert contract['measurement_only'] and contract['no_math_change'] and contract['no_candidate'] and contract['no_tolerance']
    assert json.loads((ROOT/'vae_unit138.json').read_text())['completed']
    assert json.loads((ROOT/'controller_unit138.json').read_text())['completed']
    for path,digest in fresh['native_source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path(fresh['fixed_video_latent_path']).read_bytes()).hexdigest()==fresh['fixed_video_latent_sha256']
    from cohort128 import prepare_failure_watch128,monitor_private_failure
    failure_watch=prepare_failure_watch128()
    SESSION.mkdir()
    key_path=ROOT/'private_IPC_key127.bin';assert not key_path.exists()
    fd=os.open(key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as f:f.write(os.urandom(32))
    work=ROOT/'cann_engine_vae138c';work.mkdir(exist_ok=False)
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
    shell='source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_vae13803:/tmp/h3_align12401:$PYTHONPATH; export '
    shell+=' '.join(k+'='+v for k,v in assignments.items())
    shell+='; exec /home/mseco/vllm910a/bin/python -u /tmp/h3_vae13803/generate_vae138.py'
    command=['/bin/bash','-c',shell]
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    deadline_monotonic=time.monotonic()+5400
    row={'pid':None,'launcher_pid':os.getpid(),'command':command,'source_sha256':sources,'original_source_sha256':pins,
         'compute_budget_seconds':5400,'cleanup_grace_seconds':360,'new_precision_policy_accepted':False,'model_weights_transferred':False}
    def publish():
        if receipt.exists():assert json.loads(receipt.read_text())['pid']==row['pid']
        temporary=ROOT/'engine_vae138c_process.new'
        assert temporary.resolve()==Path('/tmp/h3_vae13803/engine_vae138c_process.new')
        assert receipt.resolve()==Path('/tmp/h3_vae13803/engine_vae138c_process.json')
        temporary.write_text(json.dumps(row,indent=2)+'\n');os.replace(temporary,receipt)
    child_env={**os.environ,**assignments}
    rc=guarded_launch137(command,child_env,ROOT/'engine_vae138c.log',row,publish,failure_watch,deadline_monotonic)
    print(json.dumps({'completed_process':row['pid'],'exit_code':rc}),flush=True);raise SystemExit(rc)


if __name__=='__main__':main()
