#!/usr/bin/env python3
"""Launch the R140 roofline probe with torchrun over 8 ranks (measure only)."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/tmp/h3_fast163');LABEL='fast163b'

def main():
    process_receipt=ROOT/(LABEL+'_process.json');out=ROOT/LABEL
    assert not out.exists() and not process_receipt.exists(),'Refuse replay of a prior or uncertain launch'
    pins={'/tmp/h3_align12401/NPU_encoder_FP16_124.py':'0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026',
          '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
          '/tmp/h3_align12301/same_initial_sampling123.pt':'8a560c430b3048dd8f4ad5c7ef7d5a603cef7c4f52d6fe56aa4173e42120e3b9'}
    for path,digest in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    work=ROOT/('cann_'+LABEL);work.mkdir(exist_ok=False)
    out.mkdir(exist_ok=False)
    assignments={'H3_FAST163_LABEL':LABEL,'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT':'7200',
        'VLLM_OMNI_910A_FLASH_ATTN':'1','VLLM_OMNI_910A_FP16_SEQ':'1','VLLM_OMNI_910A_TE_DTYPE':'fp16',
        'VLLM_OMNI_910A_TEXT_HF_MATH':'1','VLLM_OMNI_910A_NO_FUSED_ACT':'1','VLLM_OMNI_910A_NO_FUSED_NORM':'1',
        'H3_NPU_NAN_COMPAT124':'1','H3_NPU_VAE_PRESERVE_FP32_PARAMS124':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1',
        'ASCEND_RT_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7','ASCEND_WORK_PATH':str(work)}
    exports=' '.join(f'{k}={v}' for k,v in assignments.items())
    command=['/bin/bash','-c',
        ('source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_fast163:/tmp/h3_align12401:$PYTHONPATH; export '+exports+
         '; exec /home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 /tmp/h3_fast163/run_fast163.py')]
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    with (ROOT/(LABEL+'.log')).open('xb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    process_receipt.write_text(json.dumps({'pid':process.pid,'command':command,'source_sha256':sources,
        'original_sources_pinned':pins,'measurement_only_no_math_change':True,'no_candidate_installed':True},indent=2)+'\n')
    print(json.dumps({'started':LABEL,'pid':process.pid}),flush=True)
    rc=process.wait()
    data=json.loads(process_receipt.read_text());data['exit_code']=rc;process_receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'completed_process':process.pid,'exit_code':rc}),flush=True)
    raise SystemExit(rc)

if __name__=='__main__':main()
