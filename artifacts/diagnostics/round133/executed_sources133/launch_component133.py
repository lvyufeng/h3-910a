#!/usr/bin/env python3
"""Bounded original-AdaLN component launcher with out-of-band exact-private abort."""
from pathlib import Path
import hashlib
import json
import os
import select
import subprocess
import threading
import time

ROOT=Path('/tmp/h3_adaln13301')


def main():
    receipt=ROOT/'component_process133.json';assert not receipt.exists(),'Never replay an uncertain small-component launch'
    assert not list(ROOT.glob('component_rank*133.json'))
    preflight=json.loads((ROOT/'preflight133.json').read_text());assert preflight['completed']
    pins=json.loads((ROOT/'source_pins133.json').read_text())['source_sha256']
    for name,digest in pins.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    from cohort128 import prepare_failure_watch128,monitor_private_failure
    from private_abort133 import private_rows133,abort_private133
    failure_fd=prepare_failure_watch128()
    shell='source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_adaln13301:/tmp/h3_align12401:$PYTHONPATH; '
    shell+='export H3_R133_PRIVATE_ROOT=/tmp/h3_adaln13301 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; '
    shell+='exec /home/mseco/vllm910a/bin/python -u -m torch.distributed.run --standalone --nproc_per_node=8 /tmp/h3_adaln13301/component133.py'
    deadline=time.monotonic()+1200
    with (ROOT/'component133.log').open('xb') as log:
        process=subprocess.Popen(['/bin/bash','-c',shell],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row={'pid':process.pid,'launcher_pid':os.getpid(),'source_sha256':pins,'start_unix':time.time(),'budget_s':1200,
        'command':['/bin/bash','-c',shell],'adopted_numeric_policy':False,'no_model_or_activation_transfer':True}
    def publish():
        temporary=ROOT/'component_process133.new';assert temporary.resolve()==Path('/tmp/h3_adaln13301/component_process133.new')
        temporary.write_text(json.dumps(row,indent=2)+'\n');os.replace(temporary,receipt)
    publish();print(json.dumps({'started':process.pid,'root':str(ROOT)}),flush=True)
    stop=threading.Event();errors=[]
    def watch():
        try:monitor_private_failure(process,stop,failure_fd,deadline_monotonic=deadline)
        except BaseException as exc:
            errors.append({'type':type(exc).__name__,'message':str(exc)})
            if not getattr(process,'_private_abort_called133',False) and private_rows133():
                try:abort_private133()
                except BaseException as error:errors.append({'type':type(error).__name__,'message':str(error)})
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    fd=os.pidfd_open(process.pid)
    try:ready,_,_=select.select([fd],[],[],max(0,deadline-time.monotonic())+360)
    finally:os.close(fd)
    if not ready:
        stop.set();watcher.join(timeout=1);row['completion_unknown']=True;row['monitor_errors']=errors;publish()
        raise TimeoutError('Original small-component teardown exceeded deadline; do not reset devices')
    rc=process.wait();stop.set();watcher.join(timeout=360)
    row['exit_code']=rc;row['end_unix']=time.time();row['monitor_errors']=errors
    if watcher.is_alive():row['completion_unknown']=True;publish();raise TimeoutError('Private component abort monitor teardown unknown')
    if private_rows133():row['private_children_cleanup']=abort_private133()
    assert not private_rows133();row['all_private_processes_exited']=True;publish()
    print(json.dumps({'completed_process':process.pid,'exit_code':rc}),flush=True);raise SystemExit(rc)


if __name__=='__main__':main()
