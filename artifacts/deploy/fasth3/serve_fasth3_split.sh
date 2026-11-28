#!/bin/bash
# FastH3 PDD-8 on ts-87: disaggregated encoder (cards 0-3) + diffusion (cards 4-7).
#
# Rationale: on 32 GB cards the encoder (63 GB), DiT (66 GB) and video VAE
# (~10 GB) cannot be resident together, and CPU offload is not the wanted
# strategy. The disaggregated topology assigns each component to its own card
# group; stage 0 runs the encoder and the VAE encoders, stage 1 runs the DiT and
# the VAE decoder, and the stages exchange the encoder conditioning payload over
# the shared-memory connector.
set -uo pipefail
source /data/h3fast/tools/h3_env.sh
export VLLM_OMNI_910A_FLASH_ATTN=1
export VLLM_OMNI_910A_TE_DTYPE=fp16
export VLLM_OMNI_910A_FP16_SEQ=1
export VLLM_OMNI_910A_TEXT_HF_MATH=1
export VLLM_OMNI_910A_NO_FUSED_ACT=1
export VLLM_OMNI_910A_NO_FUSED_NORM=1
export H3_FA215_PKG=/data/h3fast/ref/fa_pkg_v1
export H3_FA215_KERNEL_DIR=/data/h3fast/fa_kernels_fasth3
export H3_FA_BR_CAP=128
export H3_FA_BC_CAP=384
export H3_FA_TRIM_Q_LIMIT=41792
export PYTHONPATH=/data/h3fast/ref/fa_pkg_v1${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$H3_FA215_KERNEL_DIR"
exec /data/h3fast/rt/bin/vllm-omni serve --omni \
  /data/h3fast/models/FastH3-Omni-FP16 \
  --deploy-config /data/h3fast/deploy/fasth3_910a.yaml \
  --task-type ref2va \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 18091 \
  "$@"
