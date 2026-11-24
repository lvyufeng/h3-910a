#!/usr/bin/env python3
"""Bounded independent trim-only model measurement; preserve source/default behavior."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil

root=Path('/tmp/h3_trim_model218')
receipt=root/'process_trim218.json'
assert not receipt.exists() and not (root/'run_trim218.log').exists()
pins={'/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
      '/tmp/h3_align12301/same_initial_sampling123.pt':'8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9',
      '/tmp/h3_perf12601/native_sources/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'}
for path,expected in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
(root/'single_trim218').mkdir(mode=0o700)
(root/'cann').mkdir(mode=0o700)
(root/'candidate_fa_kernels').mkdir(mode=0o700)
(root/'baseline_fa_kernels').mkdir(mode=0o700)
base_key='b1_h7_n43776_d128_br128_bc304_c30_k41769_v3'
trim_key='trim218_q41792_'+base_key
for source,key,target,expected in [
    (Path('/tmp/h3_trim218/kernels/base_real'),base_key,'baseline_fa_kernels','aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8'),
    (Path('/tmp/h3_trim218/kernels/trim_real'),trim_key,'candidate_fa_kernels','052c49f2df2235bac2136b4630a05a71bd1b9b454d88c354bbbf5e254ab6e327')]:
    assert hashlib.sha256((source/key/(key+'.o')).read_bytes()).hexdigest()==expected
    shutil.copytree(source/key,root/target/key)
env=os.environ.copy()
env.update({'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200','VLLM_OMNI_910A_FLASH_ATTN':'1',
    'VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16','VLLM_OMNI_910A_TEXT_HF_MATH':'1',
    'VLLM_OMNI_910A_NO_FUSED_ACT':'1','VLLM_OMNI_910A_NO_FUSED_NORM':'1','H3_NPU_NAN_COMPAT124':'1',
    'H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7',
    'ASCEND_WORK_PATH':str(root/'cann'),'PYTHONPATH':f'{root}:/tmp/h3_align12401:'+env.get('PYTHONPATH',''),
    'H3_FA215_KERNEL_DIR':str(root/'candidate_fa_kernels'),'H3_FA_TRIM_Q_LIMIT':'41792',
    'H3_FA215_PKG':'/tmp/h3_trim218/pkg','H3_FA_BR_CAP':'128','H3_FA_BC_CAP':'304'})
for name in ('H3_FAST_AUDIT','H3_R126_IPC_KEY_FILE','H3_R126_FULL_SELECTIVE_BOOTSTRAP','VLLM_OMNI_910A_FLASH_SYNC'):env.pop(name,None)
command=['/bin/bash','-c','source /home/mseco/Ascend/cann-9.2.0/set_env.sh; exec /home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 /tmp/h3_trim_model218/run_trim_model218.py']
with (root/'run_trim218.log').open('xb') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':process.pid,'command':command,'pins':pins,'budget_seconds':1500,
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
      'single_candidate':'FA invalid padding task trimming only','all155checks':True,
      'no_numeric_tolerance_or_combo':True,'scope':'isolated eight-rank replay; not full49step pipeline'}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'started_pid':process.pid,'budget_seconds':1500}),flush=True)
try:rc=process.wait(timeout=1500)
except subprocess.TimeoutExpired:
    p=Path('/proc')/str(process.pid)
    assert (p/'cwd').resolve()==root
    cmd=(p/'cmdline').read_bytes().split(b'\0');assert any(b'/tmp/h3_trim_model218/run_trim_model218.py' in a for a in cmd)
    assert os.getpgid(process.pid)==process.pid and os.getsid(process.pid)==process.pid
    os.killpg(process.pid,signal.SIGTERM)
    try:rc=process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
    data['budget_expired']=True
finally:
    data['exit_code']=rc if 'rc' in locals() else None
    receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'exit_code':data['exit_code']}),flush=True)
raise SystemExit(data['exit_code'] or 0)
