#!/usr/bin/env python3
"""Bounded 8-rank launch with read-back receipt; no shared source changes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path('/tmp/h3_path218')
receipt=root/'process218.json'
assert not receipt.exists() and not (root/'run218.log').exists(),'Refuse uncertain launch replay'
# A pre-launch cache-selection error may have created empty directories only.
assert not (root/'path218').exists() or not any((root/'path218').iterdir()), 'Prior rank outputs exist'
pins={
    '/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
    '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
    '/tmp/h3_align12301/same_initial_sampling123.pt':'8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9',
    '/tmp/h3_budget149/matrix_safety126.py':'0d98927bf5979f5a08984dfb1cd40aa5d6e49d09b7bfa2fe8ed3be5086b30bf6',
    '/tmp/h3_perf12601/native_sources/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1',
}
for name,expected in pins.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
(root/'path218').mkdir(mode=0o700,exist_ok=True)
work=root/'cann';work.mkdir(mode=0o700,exist_ok=True)
(root/'fa_kernels').mkdir(mode=0o700,exist_ok=True)
# Select exactly one accepted binary, ignoring the cache's .build.lock file.
import shutil
key='b1_h7_n43776_d128_br128_bc304_c30_k41769_v3'
cached=Path('/tmp/h3_fa217/kernels/real_restored_end')/key
obj=cached/(key+'.o')
assert hashlib.sha256(obj.read_bytes()).hexdigest()=='aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8'
assert cached.is_dir()
destination=root/'fa_kernels'/key
if destination.exists():
    assert hashlib.sha256((destination/(key+'.o')).read_bytes()).hexdigest()=='aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8'
else:
    shutil.copytree(cached,destination)
env=os.environ.copy()
env.update({
    'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200','VLLM_OMNI_910A_FLASH_ATTN':'1',
    'VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16',
    'VLLM_OMNI_910A_TEXT_HF_MATH':'1','VLLM_OMNI_910A_NO_FUSED_ACT':'1',
    'VLLM_OMNI_910A_NO_FUSED_NORM':'1','H3_NPU_NAN_COMPAT124':'1',
    'H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7',
    'ASCEND_WORK_PATH':str(work),
    'PYTHONPATH':f'{root}:/tmp/h3_align12401:'+env.get('PYTHONPATH',''),
})
for key in ('H3_FAST_AUDIT','H3_R126_IPC_KEY_FILE','H3_R126_FULL_SELECTIVE_BOOTSTRAP','VLLM_OMNI_910A_FLASH_SYNC'):
    env.pop(key,None)
command=['/bin/bash','-c','source /home/mseco/Ascend/cann-9.2.0/set_env.sh; exec /home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 /tmp/h3_path218/run218.py']
with (root/'run218.log').open('xb') as log:
    process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':process.pid,'command':command,'pins':pins,
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
      'budget_seconds':1800,'candidate_installed':False,'live_encoder_executed':False,
      'scope':'same-process8rank DiT isolated replay TP2+USP4; all155strict checks'}
receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'started_pid':process.pid,'budget_seconds':1800}),flush=True)
try:
    rc=process.wait(timeout=1800)
except subprocess.TimeoutExpired:
    # Verify private process group identity before any signal.
    proc=Path('/proc')/str(process.pid)
    assert (proc/'cwd').resolve()==root
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert any(b'/tmp/h3_path218/run218.py' in a for a in args)
    assert os.getpgid(process.pid)==process.pid and os.getsid(process.pid)==process.pid
    os.killpg(process.pid,signal.SIGTERM)
    try:rc=process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);rc=process.wait()
    data['budget_expired']=True
finally:
    data['exit_code']=rc if 'rc' in locals() else None
    receipt.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'exit_code':data['exit_code']}),flush=True)
raise SystemExit(data['exit_code'] or 0)
