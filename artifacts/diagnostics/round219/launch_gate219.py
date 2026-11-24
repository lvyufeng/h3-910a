#!/usr/bin/env python3
"""One bounded 8rank supplemental precision gate, never candidate adoption."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil

root=Path('/tmp/h3_gate219')
receipt=root/'process219.json'
assert not receipt.exists() and not (root/'run219.log').exists()
pins={
 '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
 '/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
 '/tmp/h3_align12301/same_initial_sampling123.pt':'8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9',
 '/tmp/h3_perf12601/native_sources/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1',
 '/tmp/h3_align12401/GPU_corrected_pipeline_snapshot_step00.pt':'50ffea38a02c060b5658c76b89743373b0daf22747ba3c9dc731ed49b8857bc3',
 '/tmp/h3_align12401/GPU_corrected_pipeline_snapshot_step48.pt':'a5b8de0f21eb5c05b0d3c3e5e341bebda46329aef4529c68cfa848e6a8d78868',
}
for name,expected in pins.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256((root/'fa218trim.py').read_bytes()).hexdigest()=='52365cbf64180e9766b4dfc4c3a7bff5f311b6b8f8af7fc08b340a2bf8487b73'
for name in ('gate219','cann','candidate_fa_kernels','baseline_fa_kernels'):(root/name).mkdir(mode=0o700)
# Reuse baseline binary only; candidate recompiles from the pinned identical R218 source.
key='b1_h7_n43776_d128_br128_bc304_c30_k41769_v3'
source=Path('/tmp/h3_fa217/kernels/real_restored_end')/key
assert hashlib.sha256((source/(key+'.o')).read_bytes()).hexdigest()=='aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8'
shutil.copytree(source,root/'baseline_fa_kernels'/key)
env=os.environ.copy()
env.update({'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200','VLLM_OMNI_910A_FLASH_ATTN':'1',
 'VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16','VLLM_OMNI_910A_TEXT_HF_MATH':'1',
 'VLLM_OMNI_910A_NO_FUSED_ACT':'1','VLLM_OMNI_910A_NO_FUSED_NORM':'1','H3_NPU_NAN_COMPAT124':'1',
 'H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
 'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7',
 'ASCEND_WORK_PATH':str(root/'cann'),'PYTHONPATH':f'{root}:/tmp/h3_align12401:'+env.get('PYTHONPATH',''),
 'H3_FA215_KERNEL_DIR':str(root/'candidate_fa_kernels'),'H3_FA_TRIM_Q_LIMIT':'41792',
 'H3_FA215_PKG':str(root/'pkg'),'H3_FA_BR_CAP':'128','H3_FA_BC_CAP':'304','TMPDIR':str(root/'tmp')})
(root/'tmp').mkdir(mode=0o700)
for name in ('H3_FAST_AUDIT','H3_R126_IPC_KEY_FILE','H3_R126_FULL_SELECTIVE_BOOTSTRAP','VLLM_OMNI_910A_FLASH_SYNC'):env.pop(name,None)
command=['/bin/bash','-c','source /home/mseco/Ascend/cann-9.2.0/set_env.sh; exec /home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 /tmp/h3_gate219/run_gate219.py']
with (root/'run219.log').open('xb') as log:
 process=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
data={'pid':process.pid,'command':command,'pins':pins,'budget_seconds':1800,
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},
 'gate_only':True,'candidate_model_outputs':0,'only_original_FA_output_used':True,
 'steps':[0,48],'all_155strict_checks':True,'no_numeric_tolerance_or_combo':True,
 'live_encoder_executed':False,'scope':'isolated8rank DiT precision evidence; not49step pipeline'}
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'started_pid':process.pid,'budget_seconds':1800}),flush=True)
try:rc=process.wait(timeout=1800)
except subprocess.TimeoutExpired:
 p=Path('/proc')/str(process.pid)
 assert (p/'cwd').resolve()==root
 cmd=(p/'cmdline').read_bytes().split(b'\0');assert any(b'/tmp/h3_gate219/run_gate219.py' in a for a in cmd)
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
