#!/usr/bin/env bash
# Serve MiniMax-H3 through vLLM-Omni on the 910A fleet.
#
# The 910A generation has no usable fused attention (SDPA, aclnnFlashAttentionScore
# and PromptFlashAttention all fail), so the attention backend is pinned to the
# chunked matmul implementation registered as TORCH_SDPA.
#
# Usage:
#   bash runtime/omni-serve-910a.sh            # foreground
#   bash runtime/omni-serve-910a.sh --daemon   # background with pid/log
#
# The 910A generation is also unusable for bf16 matmul: measured 1.01 TFLOPS in
# bf16 against 170.57 TFLOPS in fp16 for the same GEMM, a 168x gap. DTYPE=half is
# therefore required, not cosmetic -- bf16 turns a 10s clip into an hour-long run.
#
# Environment knobs (defaults target 10s video / 5 min):
#   MODEL          model dir            (default /home/mseco/models/MiniMax-H3)
#   PORT           http port            (default 9198)
#   NUM_GPUS       cards                (default 8)
#   USP            sequence-parallel    (default = NUM_GPUS)
#   TE_TP          text-encoder TP size (default = NUM_GPUS)
#   LORA           distill LoRA path    (default FlashGen 4-step)
#   LORA_BACKEND   peft|distill         (default distill)
#   QUANT          json or ""           (default "" -- see below)
#   ATTN_CHUNK     910A attention chunk (default 1024)
#   HSDP           1|0 shard DiT weights (default 1)
#   DTYPE          torch dtype          (default half)
#
# QUANT defaults to off because the int8 path does not run here: vllm-omni only
# ships an fp8/HSDP quantizer upstream and 910A rejects the dynamic-quant params
# ("dynamicQuantParams.x not implemented").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/mseco/vllm910a/bin/vllm-omni}"
CANN="${CANN_HOME:-/home/mseco/Ascend/cann-9.2.0}"

MODEL="${MODEL:-/home/mseco/models/MiniMax-H3}"
PORT="${PORT:-9198}"
NUM_GPUS="${NUM_GPUS:-8}"
USP="${USP:-$NUM_GPUS}"
TE_TP="${TE_TP:-$NUM_GPUS}"
DEFAULT_LORA=/home/mseco/models/loras/flashgen_4step.safetensors
# - (not :-) so an explicit LORA="" disables the adapter
LORA="${LORA-$DEFAULT_LORA}"
LORA_BACKEND="${LORA_BACKEND:-distill}"
DEFAULT_QUANT=""
QUANT="${QUANT-$DEFAULT_QUANT}"
ATTN_CHUNK="${ATTN_CHUNK:-1024}"
STAGE_INIT_TIMEOUT="${STAGE_INIT_TIMEOUT:-2400}"
INIT_TIMEOUT="${INIT_TIMEOUT:-2400}"
DLO="${DLO:-1}"
DTYPE="${DTYPE:-half}"
TASK_TYPE="${TASK_TYPE:-fl2va}"
LOG="${LOG:-$ROOT/run/omni/serve-910a-$PORT.log}"
PIDFILE="${PIDFILE:-$ROOT/run/omni/serve-910a-$PORT.pid}"

set +u
# shellcheck disable=SC1090
source "$CANN/set_env.sh"
set -u

# CANN's AscendC kernel compiler (used to JIT built-in ops such as RmsNorm)
# does not add the system C++ include dirs, so op compile dies with
# "fatal error: 'cstdint' file not found". Point it at the toolchain.
export CPLUS_INCLUDE_PATH="/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="/usr/include:/usr/include/aarch64-linux-gnu${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"

# CANN defaults to 128 parallel fatbin compile subprocesses per op build; with
# eight diffusion workers that oversubscribes the host and the TBE subprocesses
# die ("main process disappeared"), taking the engine down with them.
export TBE_PARALLEL_COMPILER="${TBE_PARALLEL_COMPILER:-4}"

# Ascend's inductor backend inserts nan_to_num for fp16 stability, and that
# op has no kernel on 910A ("No supported Ops kernel ... optype [NanToNum]").
# Keep the graph backends off for this hardware.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE="${MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE:-0}"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# expandable_segments causes spurious "1.36 GiB allocated but OOM" failures on
# this torch_npu build, so it is opt-in rather than the default.
if [[ -n "${PYTORCH_NPU_ALLOC_CONF:-}" ]]; then
  export PYTORCH_NPU_ALLOC_CONF
fi
export VLLM_OMNI_910A_ATTN_CHUNK="$ATTN_CHUNK"

# A whole multi-step generation runs inside a single DLO collective wave, and the
# 600s default kills the run mid-generation (observed dying at step 3 of 4).
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT="${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-14400}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS-1)))}"

args=(
  serve "$MODEL"
  --omni
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --task-type "$TASK_TYPE"
  --num-gpus "$NUM_GPUS"
  --usp "$USP"
  --ring 1
  --text-encoder-tp-size "$TE_TP"
  --vae-parallel-mode tile
  --vae-use-tiling
  # The decoder splits the frame into a fixed tile count for the requested
  # resolution; if that count does not divide the rank count the whole VAE
  # falls back to decoding rank-locally (every rank decodes everything).
  --vae-patch-parallel-size "${VAE_PP:-$NUM_GPUS}"
  --diffusion-attention-backend TORCH_SDPA
  --stage-init-timeout "$STAGE_INIT_TIMEOUT"
  --init-timeout "$INIT_TIMEOUT"
  --dtype "$DTYPE"
)

# HSDP (FSDP2) shards DiT weights across ranks. Without it every rank holds a
# full weight replica, which 32GB cards cannot fit.
if [[ "${HSDP:-1}" == "1" ]]; then
  args+=(--use-hsdp --hsdp-shard-size "${HSDP_SHARD:-$NUM_GPUS}")
fi

# 910A cannot run the DLO default ALLGATHER transfer (HcclAllGatherV returns
# HCCL_E_NOT_SUPPORT and the stage init hangs), so stream weights rank-locally.
DEFAULT_OFFLOAD_CONFIG='{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"rank-local"},"text_encoder":{"weight_transfer":"rank-local"}}}'
# DLO_MODE=allgather shares one host copy across ranks (needs HCCL all_gather,
# verified working on 910A); rank-local gives each rank its own copy and costs
# ~8x host RAM, which OOMs this box at 156 GB of weights.
#
# HWR (Host Weight Runtime) is mutually exclusive with an explicit
# diffusion_offload_config -- vllm-omni rejects the combination:
#   "diffusion_offload_config cannot be combined with Host Weight Runtime"
# It drives DLO through the legacy switches instead, so HWR wins over
# OFFLOAD_CONFIG here and pins the transfer to rank-local (no AllGather).
if [[ "$DLO" == "1" ]]; then
  if [[ -n "${HWR_ROOT:-}" ]]; then
    args+=(--enable-distributed-layerwise-offload --dlo-no-use-allgather)
  elif [[ -n "${OFFLOAD_CONFIG:-}" ]]; then
    args+=(--diffusion-offload-config "$OFFLOAD_CONFIG")
  elif [[ "${DLO_MODE:-allgather}" == "allgather" ]]; then
    args+=(--enable-distributed-layerwise-offload)
  else
    args+=(--diffusion-offload-config "$DEFAULT_OFFLOAD_CONFIG")
  fi
fi
if [[ -n "$LORA" && -f "$LORA" ]]; then
  args+=(--lora-path "$LORA" --lora-backend "$LORA_BACKEND")
fi
if [[ -n "$QUANT" ]]; then
  args+=(--diffusion-quantization-config "$QUANT")
fi

# Host Weight Runtime (HWR): share ONE host copy of the offloaded weights across
# workers instead of giving each rank its own.
#
# rank-local DLO (forced on 910A, see above) flattens each rank's offloaded
# weights into a model-sized PRIVATE host allocation. With a 63 GB text encoder
# and 8 ranks that is 504 GB of host RAM, which is what exhausts this 754 GB
# box. Measured: 8 workers at ~74 GB private each (PSS == RSS, so genuinely not
# shared), total 599/603 GB private.
#
# HWR keeps the checkpoint as a node-shared mmap source instead of flattening it
# per rank. "preferred" restores an exact cache hit and otherwise loads and
# publishes normally, so it degrades gracefully rather than failing startup.
# Requires dp_size == 1 (we run usp/ring, no data parallelism).
if [[ -n "${HWR_ROOT:-}" ]]; then
  mkdir -p "$HWR_ROOT"
  args+=(
    --host-weight-runtime-mode "${HWR_MODE:-preferred}"
    --host-weight-runtime-root "$HWR_ROOT"
  )
fi

mkdir -p "$ROOT/run/omni"
echo "serving $MODEL on :$PORT  gpus=$NUM_GPUS usp=$USP te_tp=$TE_TP"
echo "lora=${LORA:-none} backend=$LORA_BACKEND quant=$QUANT attn_chunk=$ATTN_CHUNK"

if [[ "${1:-}" == "--daemon" ]]; then
  setsid nohup "$PY" "${args[@]}" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  disown || true
  echo "started pid $(cat "$PIDFILE"), log $LOG"
else
  exec "$PY" "${args[@]}"
fi