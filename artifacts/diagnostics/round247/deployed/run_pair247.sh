#!/bin/bash
# R247: run trim@128/304 then trim@64/912, one measured step each.
set -uo pipefail
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
export PYTHONPATH=/tmp/h3_budget149:/tmp/h3_budget149/trim247:/tmp/h3_align12401:${PYTHONPATH:-}
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200 VLLM_OMNI_910A_FLASH_ATTN=1 VLLM_OMNI_910A_FP16_SEQ=1
export VLLM_OMNI_910A_TE_DTYPE=fp16 VLLM_OMNI_910A_TEXT_HF_MATH=1 VLLM_OMNI_910A_NO_FUSED_ACT=1 VLLM_OMNI_910A_NO_FUSED_NORM=1
export H3_NPU_NAN_COMPAT124=1 H3_NPU_VAE_PRESERVE_FP32_PARAMS124=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_FA215_KERNEL_DIR=/tmp/fa910a_kernels
export H3_FA_TRIM_Q_LIMIT=41792
run_case () {
  local label=$1 br=$2 bc=$3
  export H3_TRIM247_LABEL=$label H3_FA_BR_CAP=$br H3_FA_BC_CAP=$bc
  export ASCEND_WORK_PATH=/tmp/h3_budget149/cann_$label
  mkdir -p /tmp/h3_budget149/$label /tmp/h3_budget149/cann_$label
  cd /tmp/h3_budget149
  echo "=== case $label br=$br bc=$bc ==="
  timeout --signal=TERM --kill-after=30s 900 /home/mseco/vllm910a/bin/torch_npu_run \
    --standalone --nproc-per-node 8 /tmp/h3_budget149/run_trim247.py \
    > /tmp/h3_budget149/$label.log 2>&1
  echo "rc=$? $label"
  grep -o '"step_s": [0-9.]*' /tmp/h3_budget149/$label.log | tail -1
  grep -o '"video_relRMS": [0-9.e-]*' /tmp/h3_budget149/$label.log | tail -1
  sleep 20
}
run_case trim247_trim128 trim=128 304 2>/dev/null || true
run_case trim247_trim128 128 304
run_case trim247_comb64 64 912
echo DONE_R247
