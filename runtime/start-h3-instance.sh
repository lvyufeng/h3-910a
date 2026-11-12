#!/usr/bin/env bash
# Foreground entry point for one isolated H3 ComfyUI process.
#
# Normally invoked by scripts/run_comfy_instance.sh, which owns instance
# validation, conflict checks and the metadata state machine.

set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${H3_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

INSTANCE_RAW="${H3_INSTANCE:-a}"
case "${INSTANCE_RAW}" in
  a|A|1)
    INSTANCE=a
    DEFAULT_DEVICES=0,1,2,3
    DEFAULT_PORT=8801
    DEFAULT_MASTER_PORT=29511
    DEFAULT_HCCL_RANGE=60000-60999
    ;;
  b|B|2)
    INSTANCE=b
    DEFAULT_DEVICES=4,5,6,7
    DEFAULT_PORT=8802
    DEFAULT_MASTER_PORT=29512
    DEFAULT_HCCL_RANGE=61000-61999
    ;;
  *)
    die "H3_INSTANCE must be a/A/1 or b/B/2, got ${INSTANCE_RAW}"
    ;;
esac

# Legacy direct-launch flags are accepted for compatibility. The control
# script passes the equivalent H3_* environment variables.
while (($#)); do
  case "$1" in
    --npu-devices)
      (($# >= 2)) || die "--npu-devices requires a value"
      H3_NPU_DEVICES="$2"
      shift 2
      ;;
    --port)
      (($# >= 2)) || die "--port requires a value"
      H3_PORT="$2"
      shift 2
      ;;
    --master-port)
      (($# >= 2)) || die "--master-port requires a value"
      H3_HCCL_MASTER_PORT="$2"
      shift 2
      ;;
    --master-port-range)
      (($# >= 2)) || die "--master-port-range requires a value"
      H3_HCCL_PORT_RANGE="$2"
      shift 2
      ;;
    --master-addr)
      (($# >= 2)) || die "--master-addr requires a value"
      H3_MASTER_ADDR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

NPU_DEVICES="${H3_NPU_DEVICES:-${DEFAULT_DEVICES}}"
COMFYUI_PORT="${H3_PORT:-${DEFAULT_PORT}}"
H3_MASTER_PORT="${H3_HCCL_MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
HCCL_NPU_SOCKET_PORT_RANGE="${H3_HCCL_PORT_RANGE:-${DEFAULT_HCCL_RANGE}}"
MASTER_ADDR_VALUE="${H3_MASTER_ADDR:-127.0.0.1}"
PYTHON_BIN="${H3_PYTHON:-${PYTHON_BIN:-${ROOT}/envs/py312/bin/python}}"
COMFYUI_ROOT="${H3_COMFY_HOME:-${COMFY_HOME:-${ROOT}/upstream/ComfyUI}}"
CANN_HOME="${H3_CANN_HOME:-${ASCEND_HOME_PATH:-/home/mseco/Ascend/cann-9.2.0}}"
CANN_ENV="${H3_CANN_ENV:-${CANN_HOME}/set_env.sh}"
EXTRA_ENV_FILE="${H3_EXTRA_ENV_FILE:-}"
RUN_ROOT="${H3_RUN_ROOT:-${ROOT}/run/h3-${INSTANCE}}"
LOG_ROOT="${H3_LOG_ROOT:-${ROOT}/logs/h3-${INSTANCE}}"
DEFAULT_MODEL_CONFIG="${ROOT}/extra_model_paths.yaml"
if [[ ! -f "${DEFAULT_MODEL_CONFIG}" ]]; then
  DEFAULT_MODEL_CONFIG="${ROOT}/config/extra_model_paths.yaml"
fi
MODEL_CONFIG_SOURCE="${H3_MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}"
METADATA_FILE="${H3_INSTANCE_METADATA:-${RUN_ROOT}/instance.json}"
PID_FILE="${RUN_ROOT}/comfyui.pid"
DB_FILE="${RUN_ROOT}/comfyui.db"
MODEL_CONFIG="${RUN_ROOT}/extra_model_paths.yaml"
CONTROL_PYTHON="${H3_CONTROL_PYTHON:-python3}"

[[ -x "${PYTHON_BIN}" ]] || die "H3_PYTHON is not executable: ${PYTHON_BIN}"
[[ -f "${COMFYUI_ROOT}/main.py" ]] || die "missing ${COMFYUI_ROOT}/main.py"
[[ -f "${CANN_ENV}" ]] || die "missing CANN environment file: ${CANN_ENV}"
[[ -f "${MODEL_CONFIG_SOURCE}" ]] ||
  die "missing model path config: ${MODEL_CONFIG_SOURCE}"
if [[ -n "${EXTRA_ENV_FILE}" && ! -f "${EXTRA_ENV_FILE}" ]]; then
  die "missing H3_EXTRA_ENV_FILE: ${EXTRA_ENV_FILE}"
fi
command -v "${CONTROL_PYTHON}" >/dev/null 2>&1 ||
  die "H3_CONTROL_PYTHON not found: ${CONTROL_PYTHON}"

if [[ -z "${H3_RUN_TOKEN:-}" ]]; then
  H3_RUN_TOKEN="$(od -An -N16 -tx1 /dev/urandom | tr -d '[:space:]')"
  [[ "${H3_RUN_TOKEN}" =~ ^[0-9a-f]{32}$ ]] ||
    die "failed to generate a run token"
fi
export H3_RUN_TOKEN
export H3_INSTANCE_METADATA="${METADATA_FILE}"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] &&
    kill -0 "${old_pid}" 2>/dev/null; then
    die "instance ${INSTANCE} is already running as PID ${old_pid}"
  fi
  die "stale PID file exists at ${PID_FILE}; preserve it for inspection"
fi

mkdir -p \
  "${RUN_ROOT}"/{user,input,output,temp,tmp,cache,ascend-log,workflows} \
  "${LOG_ROOT}"
cp -f "${MODEL_CONFIG_SOURCE}" "${MODEL_CONFIG}"

set +u
# shellcheck disable=SC1090,SC1091
source "${CANN_ENV}"
if [[ -n "${EXTRA_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${EXTRA_ENV_FILE}"
fi
set -u

# Isolation settings intentionally win over values sourced above.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export H3_INSTANCE="${INSTANCE}"
export H3_CANN_HOME="${CANN_HOME}"
export H3_CANN_ENV="${CANN_ENV}"
export ASCEND_HOME_PATH="${CANN_HOME}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export MASTER_ADDR="${MASTER_ADDR_VALUE}"
export MASTER_PORT="${H3_MASTER_PORT}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export COMFYUI_MULTI_DEVICE_BACKEND="hccl"
unset COMFYUI_TEXT_TP_BACKEND
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export H3_910A_QUANT_COMPAT="${H3_910A_QUANT_COMPAT:-1}"
export H3_910A_QUANT_CHUNK="${H3_910A_QUANT_CHUNK:-1024}"
export ASCEND_PROCESS_LOG_PATH="${RUN_ROOT}/ascend-log"
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/cache/pycache"
export TMPDIR="${RUN_ROOT}/tmp"
export TMP="${RUN_ROOT}/tmp"
export TEMP="${RUN_ROOT}/tmp"
export XDG_CACHE_HOME="${RUN_ROOT}/cache"
export HF_HOME="${RUN_ROOT}/cache/huggingface"
export TORCH_HOME="${RUN_ROOT}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${RUN_ROOT}/cache/torchinductor"
export TRITON_CACHE_DIR="${RUN_ROOT}/cache/triton"
export PYTHONPATH="${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
if [[ -n "${H3_LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="${H3_LD_PRELOAD}"
fi

echo "$$" > "${PID_FILE}"
printf 'instance=%s pid=%s devices=%s port=%s master_addr=%s master_port=%s hccl_ports=%s backend=%s run_root=%s\n' \
  "${INSTANCE}" "$$" "${NPU_DEVICES}" "${COMFYUI_PORT}" "${MASTER_ADDR_VALUE}" \
  "${H3_MASTER_PORT}" "${HCCL_NPU_SOCKET_PORT_RANGE}" \
  "${COMFYUI_MULTI_DEVICE_BACKEND}" "${RUN_ROOT}" \
  > "${RUN_ROOT}/instance.env"

"${CONTROL_PYTHON}" - \
  "${METADATA_FILE}" \
  "${H3_RUN_TOKEN}" \
  "$$" \
  "${INSTANCE}" \
  "running" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

target, token, pid_text, instance, state = sys.argv[1:]
path = Path(target)
data = {}
if path.is_file():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot update invalid instance metadata: {exc}")
if data.get("token") != token or data.get("instance") != instance:
    raise SystemExit("instance metadata identity does not match launcher")
data["pid"] = int(pid_text)
data["state"] = state
data["updated_at"] = datetime.now(timezone.utc).isoformat()
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY

cd "${COMFYUI_ROOT}"
exec "${PYTHON_BIN}" main.py \
  --listen 0.0.0.0 \
  --port "${COMFYUI_PORT}" \
  --disable-auto-launch \
  --cache-classic \
  --database-url "sqlite:///${DB_FILE}" \
  --user-directory "${RUN_ROOT}/user" \
  --input-directory "${RUN_ROOT}/input" \
  --output-directory "${RUN_ROOT}/output" \
  --temp-directory "${RUN_ROOT}/temp" \
  --extra-model-paths-config "${MODEL_CONFIG}" \
  "$@"
