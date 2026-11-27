#!/bin/bash
# MiniMax-H3 FL2VA service launcher for 8x Ascend 910A.
#
# Mirrors the validated R126 runtime contract (same bootstrap, same vllm-omni
# 910A flags, same FP16-matrix / FP32-exception alignment) and adds the measured
# speedups through the deployment sitecustomize:
#   * FA V1 tiles/pipeline (user-approved, relRMS 1.29e-4 / 1 FP16 ulp)
#   * guard fast path and fp32-out GEMMs (both bit-identical)
#
# The deploy directory is placed FIRST on PYTHONPATH so its sitecustomize is the
# one every engine worker imports.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF_ROOT="${H3_REF_ROOT:-/tmp/h3_budget149}"      # verified bootstrap + fixtures
ALIGN_ROOT="${H3_ALIGN_ROOT:-/tmp/h3_align12401}"  # verified model replay + fixtures
PY="${H3_PYTHON:-/home/mseco/vllm910a/bin/python}"
CANN_ENV="${H3_CANN_ENV:-/home/mseco/Ascend/cann-9.2.0/set_env.sh}"

MODEL="${H3_MODEL:-${ALIGN_ROOT}/original_metadata124f/FL2VA}"
OUTPUT_ROOT="${H3_OUTPUT_ROOT:-/var/lib/h3-service/outputs}"
PORT="${H3_PORT:-8000}"
HOST="${H3_HOST:-0.0.0.0}"
WORK="${H3_WORK:-/tmp/h3-service/work}"

# shellcheck disable=SC1090
source "$CANN_ENV"
mkdir -p "$OUTPUT_ROOT" "$WORK"

# Private IPC credential shared by every engine worker (the reference contract).
KEY_FILE="${H3_R126_IPC_KEY_FILE:-${REF_ROOT}/private_IPC_key126.bin}"
if [ ! -f "$KEY_FILE" ]; then
  umask 077
  head -c 32 /dev/urandom > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

export PYTHONPATH="${HERE}:${REF_ROOT}/vae:${REF_ROOT}:${ALIGN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# --- validated R126 runtime environment -------------------------------------
export H3_R126_IPC_KEY_FILE="$KEY_FILE"
export H3_R126_FULL_SELECTIVE_BOOTSTRAP=1
export H3_R126_VAE_OPERAND_CACHE=1
export H3_NPU_ALIGNMENT124_BOOTSTRAP=0
export H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP=0
# integration_hooks126 requires this private capture root to live under the reference
# checkout and to carry the generation label; service media goes to $OUTPUT_ROOT instead.
GENERATION_DIR="${H3_R126_GENERATION_DIR:-${REF_ROOT}/NPU_full_generation126_service}"
mkdir -p "$GENERATION_DIR"
export H3_R126_GENERATION_OUT="$GENERATION_DIR"
export H3_NPU_RAW_OUTPUT_DIR124="$GENERATION_DIR"
export H3_NPU_CAPTURE_NATIVE_MEDIA124="${H3_NPU_CAPTURE_NATIVE_MEDIA124:-0}"
export H3_NPU_SAVE_FINAL_LATENTS124="${H3_NPU_SAVE_FINAL_LATENTS124:-0}"
export H3_NPU_NAN_COMPAT124=1
export H3_NPU_VAE_PRESERVE_FP32_PARAMS124=1

export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200
export VLLM_OMNI_910A_FLASH_ATTN=1
export VLLM_OMNI_910A_FP16_SEQ=1
export VLLM_OMNI_910A_TE_DTYPE=fp16
export VLLM_OMNI_910A_TEXT_HF_MATH=1
export VLLM_OMNI_910A_NO_FUSED_ACT=1
export VLLM_OMNI_910A_NO_FUSED_NORM=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_WORK_PATH="$WORK"

# --- deployment speedups ----------------------------------------------------
export H3_SERVICE_FA_V1="${H3_SERVICE_FA_V1:-1}"
export H3_SERVICE_GUARD_FAST="${H3_SERVICE_GUARD_FAST:-1}"
export H3_SERVICE_FAST_GEMM="${H3_SERVICE_FAST_GEMM:-1}"
# V1 kernels must live in their own cache: the cache identity includes the package.
export H3_FA215_KERNEL_DIR="${H3_FA215_KERNEL_DIR:-/tmp/fa_kernels_v1}"
export H3_FA215_PKG="${H3_FA215_PKG:-/tmp/fa_pkg_v1}"
export H3_FA_BR_CAP="${H3_FA_BR_CAP:-128}"
export H3_FA_BC_CAP="${H3_FA_BC_CAP:-384}"
export H3_FA_TRIM_Q_LIMIT="${H3_FA_TRIM_Q_LIMIT:-41792}"

echo "[serve_h3] here=$HERE"
echo "[serve_h3] model=$MODEL"
echo "[serve_h3] output_root=$OUTPUT_ROOT http=$HOST:$PORT"
echo "[serve_h3] fa_v1=$H3_SERVICE_FA_V1 guard_fast=$H3_SERVICE_GUARD_FAST fast_gemm=$H3_SERVICE_FAST_GEMM"

exec "$PY" -u "$HERE/h3_service.py" \
  --model "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --host "$HOST" \
  --port "$PORT" \
  "$@"