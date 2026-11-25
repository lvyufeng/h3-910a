"""Launch the official full-generation script exactly like R126, with the lazy stage probe injected."""
import os
import shlex
import subprocess
import sys
from pathlib import Path

LABEL = sys.argv[1]
SKIP = os.environ.get('H3_SKIP_ENCODER_OFFLOAD', '0')
WORK = Path('/tmp/h3_r221/cann_' + LABEL)
OUT = Path('/tmp/h3_budget149/' + LABEL)
WORK.mkdir(parents=True, exist_ok=True)

assignments = {
    'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT': '7200',
    'VLLM_OMNI_910A_FLASH_ATTN': '1', 'VLLM_OMNI_910A_FP16_SEQ': '1', 'VLLM_OMNI_910A_TE_DTYPE': 'fp16',
    'VLLM_OMNI_910A_TEXT_HF_MATH': '1', 'VLLM_OMNI_910A_NO_FUSED_ACT': '1', 'VLLM_OMNI_910A_NO_FUSED_NORM': '1',
    'H3_NPU_NAN_COMPAT124': '1', 'H3_NPU_VAE_PRESERVE_FP32_PARAMS124': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
    'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'ASCEND_WORK_PATH': str(WORK),
    'H3_STAGE_PROBE': '1', 'H3_STAGE_OUT': '/tmp/h3_r221/stage_times_%s.json' % LABEL,
    'H3_SKIP_ENCODER_OFFLOAD': SKIP,
}
# keep the inherited PYTHONPATH first so /tmp/h3_r221 (sitecustomize probe) is importable
exports = ' '.join('%s=%s' % (k, shlex.quote(v)) for k, v in assignments.items())
cmd = ['/bin/bash', '-c',
       'source /home/mseco/Ascend/cann-9.2.0/set_env.sh; '
       'export PYTHONPATH=/tmp/h3_budget149:/tmp/h3_align12401:/tmp/h3_r221:$PYTHONPATH; '
       'export ' + exports + '; '
       'exec /home/mseco/vllm910a/bin/python -u '
       '/tmp/h3_budget149/generate_full_NPU126.py --output-label ' + LABEL]
log = Path('/tmp/h3_r221/%s.log' % LABEL)
with log.open('wb') as f:
    proc = subprocess.Popen(cmd, cwd='/tmp/h3_budget149', stdin=subprocess.DEVNULL,
                            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
Path('/tmp/h3_r221/%s.pid' % LABEL).write_text(str(proc.pid))
print('started', proc.pid, flush=True)
print('exit', proc.wait(), flush=True)
