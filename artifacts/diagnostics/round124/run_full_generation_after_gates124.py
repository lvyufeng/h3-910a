#!/usr/bin/env python3
"""Launch full original NPU generation only after the requested sigma gates complete.

Waits on the exact recorded gate PID using Linux pidfd, not an SSH/status polling
loop. Reconciles every gate receipt and refuses if any check failed. All child logs
are saved remotely; transport loss cannot restart or duplicate the generation.
"""
import json
import os
from pathlib import Path
import select
import subprocess

ROOT=Path('/tmp/h3_align12401')


def main():
    receipt=ROOT/'full_generation_supervisor124.json';assert not receipt.exists()
    waiter_receipt=ROOT/'full_generation_waiter124.json';assert not waiter_receipt.exists()
    waiter_receipt.write_text(json.dumps({'supervisor_pid':os.getpid(),'phase':'waiting_for_sigma_gates','full_generation_started':False},indent=2)+'\n')
    gate=json.loads((ROOT/'NPU_sigma_checks_process124.json').read_text());pid=gate['pid']
    proc=Path('/proc')/str(pid)
    if proc.exists():
        assert b'check_NPU_representative_sigmas124.py' in (proc/'cmdline').read_bytes()
        fd=os.pidfd_open(pid);poll=select.poll();poll.register(fd,select.POLLIN)
        print(json.dumps({'awaiting_exact_sigma_gate_process':pid,'no_full_generation_started_yet':True}),flush=True)
        poll.poll();os.close(fd)
    paths=list((ROOT/'NPU_sigma_checks124').glob('sigma_checks_rank*.json'))
    assert len(paths)==8 and all(json.loads(path.read_text())['completed'] for path in paths)
    gates=json.loads((ROOT/'NPU_sigma_checks124/sigma_checks_rank0.json').read_text())
    assert gates['all_selected_sigma_gates_accepted'] and {x['step_index'] for x in gates['checks']}=={0,10,25,40,48}
    assert all(x['video_target_velocity']['accepted_under_1percent_gate'] and x['audio_velocity']['accepted_under_1percent_gate'] for x in gates['checks'])
    env=os.environ.copy();env.update({'PYTHONPATH':str(ROOT)+':'+env.get('PYTHONPATH',''),'H3_NPU_ALIGNMENT124_BOOTSTRAP':'1',
                 'VLLM_OMNI_910A_TE_DTYPE':'fp16','VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TEXT_HF_MATH':'1',
                 'HF_HUB_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
    command=['/home/mseco/vllm910a/bin/python','-u',str(ROOT/'generate_full_NPU124.py')]
    with (ROOT/'NPU_full_generation124.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':process.pid,'group':process.pid,'command':command,'all5_same_input_sigma_gates_accepted_before_launch':True,'own_NPU_encoder_and_VAE_required':True}
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    waiter_receipt.write_text(json.dumps({'supervisor_pid':os.getpid(),'phase':'full_generation_running','full_generation_started':True,'generation_pid':process.pid},indent=2)+'\n')
    print(json.dumps({'NPU_full_generation_started_after_gates':True,'pid':process.pid}),flush=True)
    rc=process.wait()
    record['exit_code']=rc
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'NPU_full_generation_exit_code':rc}),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
