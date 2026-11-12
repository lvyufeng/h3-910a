#!/usr/bin/env bash
# Start and control one isolated MiniMax H3 ComfyUI instance.
#
# Usage:
#   bash scripts/run_comfy_instance.sh start [a|b|1|2] [-- COMFY_ARGS...]
#   bash scripts/run_comfy_instance.sh foreground [a|b|1|2] [-- COMFY_ARGS...]
#   bash scripts/run_comfy_instance.sh status [a|b|1|2]
#   bash scripts/run_comfy_instance.sh logs [a|b|1|2]
#   bash scripts/run_comfy_instance.sh stop [a|b|1|2]
#
# H3_INSTANCE_ID=1|2 selects the instance when no positional instance is given.
# This script only starts user-space ComfyUI processes. It does not change
# driver, CANN, network, sysctl, Ascend, or other system settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${H3_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
START_SCRIPT="${SCRIPT_DIR}/../runtime/start-h3-instance.sh"
CONTROL_PYTHON="${H3_CONTROL_PYTHON:-python3}"
COMMAND="${1:-}"

usage() {
  cat <<'EOF'
Usage:
  run_comfy_instance.sh {start|foreground|status|logs|stop} [a|b|1|2] [-- COMFY_ARGS...]

Instance selection:
  H3_INSTANCE_ID=1|2 (default: 1); positional a/b/1/2 overrides it.

Defaults:
  A: NPU 0,1,2,3, HTTP 8801, rendezvous 29511, socket range 60000-60999
  B: NPU 4,5,6,7, HTTP 8802, rendezvous 29512, socket range 61000-61999

Important overrides:
  H3_NPU_DEVICES H3_PORT H3_HCCL_MASTER_PORT H3_HCCL_PORT_RANGE
  H3_MASTER_ADDR H3_INSTANCE_ROOT H3_RUN_BASE H3_LOG_BASE
  H3_RUN_ROOT H3_LOG_ROOT H3_PEER_RUN_ROOT H3_PEER_LOG_ROOT
  H3_CONTROL_LOCK_DIR H3_COMFY_HOME H3_PYTHON H3_CANN_HOME
  H3_CANN_ENV H3_MODEL_CONFIG H3_EXTRA_ENV_FILE H3_LD_PRELOAD

ComfyUI arguments after "--" are passed unchanged. Arguments controlled by this
launcher are rejected to keep ports and writable paths isolated.
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

case "${COMMAND}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  start|foreground|status|logs|stop) ;;
  "")
    usage >&2
    exit 2
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
shift

INSTANCE_RAW="${H3_INSTANCE_ID:-${H3_INSTANCE:-1}}"
if (($#)) && [[ "${1}" =~ ^(a|A|b|B|1|2)$ ]]; then
  INSTANCE_RAW="$1"
  shift
fi

COMFY_ARGS=()
if (($#)); then
  if [[ "${1}" != "--" ]]; then
    echo "unexpected argument: ${1}" >&2
    usage >&2
    exit 2
  fi
  shift
  COMFY_ARGS=("$@")
fi

case "${INSTANCE_RAW}" in
  1|a|A)
    INSTANCE_ID=1
    INSTANCE_LABEL=a
    PEER_LABEL=b
    DEFAULT_DEVICES=0,1,2,3
    DEFAULT_PORT=8801
    DEFAULT_MASTER_PORT=29511
    DEFAULT_HCCL_RANGE=60000-60999
    ;;
  2|b|B)
    INSTANCE_ID=2
    INSTANCE_LABEL=b
    PEER_LABEL=a
    DEFAULT_DEVICES=4,5,6,7
    DEFAULT_PORT=8802
    DEFAULT_MASTER_PORT=29512
    DEFAULT_HCCL_RANGE=61000-61999
    ;;
  *)
    die "instance must be a/b/1/2, got ${INSTANCE_RAW}"
    ;;
esac

if ! command -v "${CONTROL_PYTHON}" >/dev/null 2>&1; then
  die "H3_CONTROL_PYTHON not found: ${CONTROL_PYTHON}"
fi
if [[ ! -f "${START_SCRIPT}" ]]; then
  die "missing foreground launcher: ${START_SCRIPT}"
fi
if [[ "${COMMAND}" != "status" && "${COMMAND}" != "logs" && "${COMMAND}" != "stop" ]]; then
  if ! command -v curl >/dev/null 2>&1; then
    die "curl is required for readiness checks"
  fi
fi

DEVICES="${H3_NPU_DEVICES:-${DEFAULT_DEVICES}}"
PORT="${H3_PORT:-${DEFAULT_PORT}}"
HCCL_MASTER_PORT="${H3_HCCL_MASTER_PORT:-${DEFAULT_MASTER_PORT}}"
HCCL_PORT_RANGE="${H3_HCCL_PORT_RANGE:-${DEFAULT_HCCL_RANGE}}"
MASTER_ADDR="${H3_MASTER_ADDR:-127.0.0.1}"
EXPECTED_DEVICE_COUNT="${H3_EXPECTED_DEVICE_COUNT:-4}"
AVAILABLE_DEVICE_COUNT="${H3_AVAILABLE_DEVICE_COUNT:-8}"
START_TIMEOUT="${H3_START_TIMEOUT:-300}"
STOP_TIMEOUT="${H3_STOP_TIMEOUT:-60}"
HEALTH_HOST="${H3_HEALTH_HOST:-127.0.0.1}"
COMFY_HOME="${H3_COMFY_HOME:-${COMFY_HOME:-${ROOT}/upstream/ComfyUI}}"
PYTHON_BIN="${H3_PYTHON:-${ROOT}/envs/py312/bin/python}"
CANN_HOME="${H3_CANN_HOME:-${ASCEND_HOME_PATH:-/home/mseco/Ascend/cann-9.2.0}}"
CANN_ENV="${H3_CANN_ENV:-${CANN_HOME}/set_env.sh}"
DEFAULT_MODEL_CONFIG="${ROOT}/extra_model_paths.yaml"
if [[ ! -f "${DEFAULT_MODEL_CONFIG}" ]]; then
  DEFAULT_MODEL_CONFIG="${ROOT}/config/extra_model_paths.yaml"
fi
MODEL_CONFIG="${H3_MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}"
EXTRA_ENV_FILE="${H3_EXTRA_ENV_FILE:-}"
LD_PRELOAD_VALUE="${H3_LD_PRELOAD:-}"
MULTI_DEVICE_BACKEND="hccl"

INSTANCE_ROOT="${H3_INSTANCE_ROOT:-}"
if [[ -n "${INSTANCE_ROOT}" ]]; then
  RUN_BASE="${H3_RUN_BASE:-${INSTANCE_ROOT}}"
  LOG_BASE="${H3_LOG_BASE:-${INSTANCE_ROOT}/logs}"
  DEFAULT_CONTROL_LOCK_DIR="${INSTANCE_ROOT}/.h3-instance-control.lock"
else
  RUN_BASE="${H3_RUN_BASE:-${ROOT}/run}"
  LOG_BASE="${H3_LOG_BASE:-${ROOT}/logs}"
  DEFAULT_CONTROL_LOCK_DIR="${RUN_BASE}/.h3-instance-control.lock"
fi
PEER_RUN_BASE="${H3_PEER_INSTANCE_ROOT:-${RUN_BASE}}"
PEER_LOG_BASE="${H3_PEER_LOG_BASE:-${LOG_BASE}}"
DEFAULT_RUN_ROOT="${RUN_BASE}/h3-${INSTANCE_LABEL}"
DEFAULT_PEER_RUN_ROOT="${PEER_RUN_BASE}/h3-${PEER_LABEL}"
RUN_ROOT="${H3_RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
PEER_RUN_ROOT="${H3_PEER_RUN_ROOT:-${DEFAULT_PEER_RUN_ROOT}}"
DEFAULT_LOG_ROOT="${LOG_BASE}/h3-${INSTANCE_LABEL}"
DEFAULT_PEER_LOG_ROOT="${PEER_LOG_BASE}/h3-${PEER_LABEL}"
LOG_ROOT="${H3_LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
PEER_LOG_ROOT="${H3_PEER_LOG_ROOT:-${DEFAULT_PEER_LOG_ROOT}}"
CONTROL_LOCK_DIR="${H3_CONTROL_LOCK_DIR:-${DEFAULT_CONTROL_LOCK_DIR}}"

validate_no_record_separators() {
  local name="$1"
  local value="$2"
  if [[ "${value}" == *$'\n'* || "${value}" == *$'\t'* ]]; then
    die "${name} must not contain a newline or tab"
  fi
}

normalize_path() {
  local raw="$1"
  "${CONTROL_PYTHON}" -c \
    'import os, sys; print(os.path.realpath(os.path.abspath(sys.argv[1])))' \
    "${raw}"
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    die "${name} must be a positive integer, got ${value}"
  fi
}

validate_port() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || ((value < 1 || value > 65535)); then
    die "${name} must be a TCP port in 1..65535, got ${value}"
  fi
}

validate_port_range() {
  local name="$1"
  local value="$2"
  local start end
  if [[ ! "${value}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    die "${name} must be START-END, got ${value}"
  fi
  start="${BASH_REMATCH[1]}"
  end="${BASH_REMATCH[2]}"
  if ((start < 1 || end > 65535 || start > end)); then
    die "${name} must satisfy 1 <= START <= END <= 65535, got ${value}"
  fi
  HCCL_RANGE_START="${start}"
  HCCL_RANGE_END="${end}"
}

validate_devices() {
  local value="$1"
  local device
  local -a items=()
  local -A seen=()
  if [[ -z "${value}" ]]; then
    die "H3_NPU_DEVICES must not be empty"
  fi
  IFS=',' read -r -a items <<< "${value}"
  if ((${#items[@]} != EXPECTED_DEVICE_COUNT)); then
    die "H3_NPU_DEVICES has ${#items[@]} entries, expected ${EXPECTED_DEVICE_COUNT}"
  fi
  for device in "${items[@]}"; do
    if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
      die "invalid NPU device ID in H3_NPU_DEVICES: ${value}"
    fi
    if ((device >= AVAILABLE_DEVICE_COUNT)); then
      die "NPU device ${device} is outside 0..$((AVAILABLE_DEVICE_COUNT - 1))"
    fi
    if [[ -n "${seen[${device}]:-}" ]]; then
      die "duplicate NPU device ${device} in H3_NPU_DEVICES=${value}"
    fi
    seen["${device}"]=1
  done
}

paths_overlap() {
  local left="$1"
  local right="$2"
  [[ "${left}" == "${right}" ||
    "${left}" == "${right}/"* ||
    "${right}" == "${left}/"* ]]
}

devices_overlap() {
  "${CONTROL_PYTHON}" - "$1" "$2" <<'PY'
import sys
left = {part for part in sys.argv[1].split(",") if part}
right = {part for part in sys.argv[2].split(",") if part}
raise SystemExit(0 if left & right else 1)
PY
}

ranges_overlap() {
  local left_start="$1"
  local left_end="$2"
  local right_start="$3"
  local right_end="$4"
  ((left_start <= right_end && right_start <= left_end))
}

port_in_range() {
  local port="$1"
  local start="$2"
  local end="$3"
  ((port >= start && port <= end))
}

pid_has_instance_identity() {
  local pid="$1"
  local token="$2"
  local label="$3"
  local environ_path="/proc/${pid}/environ"
  if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] ||
    [[ ! -r "${environ_path}" ]] ||
    ! kill -0 "${pid}" 2>/dev/null; then
    return 1
  fi
  tr '\0' '\n' < "${environ_path}" |
    grep -Fxq "H3_RUN_TOKEN=${token}" || return 1
  tr '\0' '\n' < "${environ_path}" |
    grep -Fxq "H3_INSTANCE=${label}" || return 1
}

metadata_dump() {
  local path="$1"
  "${CONTROL_PYTHON}" - "${path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as source:
    data = json.load(source)

fields = (
    "token",
    "pid",
    "devices",
    "port",
    "hccl_master_port",
    "hccl_port_range",
    "run_root",
    "log_root",
    "multi_device_backend",
    "state",
)
values = []
for field in fields:
    value = data.get(field)
    if value is None:
        value = ""
    text = str(value)
    if "\t" in text or "\n" in text or "\r" in text:
        raise SystemExit(f"unsafe metadata field {field}")
    values.append(text)
print("\t".join(values))
PY
}

read_metadata() {
  local path="$1"
  local prefix="$2"
  local dump
  if [[ ! -f "${path}" ]]; then
    return 1
  fi
  if ! dump="$(metadata_dump "${path}")"; then
    die "cannot parse instance metadata: ${path}"
  fi
  IFS=$'\t' read -r \
    "${prefix}_token" "${prefix}_pid" "${prefix}_devices" \
    "${prefix}_port" "${prefix}_hccl_master" "${prefix}_hccl_range" \
    "${prefix}_run_root" "${prefix}_log_root" \
    "${prefix}_multi_device_backend" "${prefix}_state" <<< "${dump}"
}

write_metadata() {
  local state="$1"
  local pid="${2:-}"
  "${CONTROL_PYTHON}" - \
    "${METADATA_FILE}" \
    "${H3_RUN_TOKEN}" \
    "${pid}" \
    "${INSTANCE_ID}" \
    "${INSTANCE_LABEL}" \
    "${DEVICES}" \
    "${PORT}" \
    "${HCCL_MASTER_PORT}" \
    "${HCCL_PORT_RANGE}" \
    "${MASTER_ADDR}" \
    "${RUN_ROOT}" \
    "${LOG_ROOT}" \
    "${COMFY_HOME}" \
    "${MULTI_DEVICE_BACKEND}" \
    "${state}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    target,
    token,
    pid_text,
    instance_id,
    instance_label,
    devices,
    port,
    hccl_master_port,
    hccl_port_range,
    master_addr,
    run_root,
    log_root,
    comfy_home,
    multi_device_backend,
    state,
) = sys.argv[1:]

target_path = Path(target)
created_at = None
if target_path.is_file():
    try:
        old = json.loads(target_path.read_text(encoding="utf-8"))
        created_at = old.get("created_at")
    except (OSError, json.JSONDecodeError):
        pass

now = datetime.now(timezone.utc).isoformat()
payload = {
    "schema": 1,
    "instance_id": int(instance_id),
    "instance": instance_label,
    "state": state,
    "token": token,
    "pid": int(pid_text) if pid_text else None,
    "devices": devices,
    "port": int(port),
    "hccl_master_port": int(hccl_master_port),
    "hccl_port_range": hccl_port_range,
    "master_addr": master_addr,
    "run_root": run_root,
    "log_root": log_root,
    "comfy_home": comfy_home,
    "multi_device_backend": multi_device_backend,
    "created_at": created_at or now,
    "updated_at": now,
}
temporary = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target_path)
PY
}

requested_config_matches_metadata() {
  local prefix="$1"
  local meta_devices="$2"
  local meta_port="$3"
  local meta_master="$4"
  local meta_range="$5"
  local meta_run="$6"
  local meta_log="$7"
  if [[ "${meta_devices}" != "${DEVICES}" ||
    "${meta_port}" != "${PORT}" ||
    "${meta_master}" != "${HCCL_MASTER_PORT}" ||
    "${meta_range}" != "${HCCL_PORT_RANGE}" ||
    "$(normalize_path "${meta_run}")" != "${RUN_ROOT}" ||
    "$(normalize_path "${meta_log}")" != "${LOG_ROOT}" ]]; then
    echo "${prefix} configuration differs from the requested configuration" >&2
    return 1
  fi
}

check_peer_conflicts() {
  local peer_meta="${PEER_RUN_ROOT}/instance.json"
  local peer_label="${PEER_LABEL}"
  local PEER_token PEER_pid PEER_devices PEER_port PEER_hccl_master
  local PEER_hccl_range PEER_run_root PEER_log_root
  if [[ ! -f "${peer_meta}" ]]; then
    return 0
  fi
  read_metadata "${peer_meta}" PEER
  if [[ ! "${PEER_pid}" =~ ^[1-9][0-9]*$ ]] ||
    ! pid_has_instance_identity "${PEER_pid}" "${PEER_token}" "${peer_label}"; then
    return 0
  fi

  local peer_run peer_log peer_range_start peer_range_end
  peer_run="$(normalize_path "${PEER_run_root}")"
  peer_log="$(normalize_path "${PEER_log_root}")"
  if [[ ! "${PEER_hccl_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    die "active peer ${peer_label} has invalid HCCL range: ${PEER_hccl_range}"
  fi
  peer_range_start="${BASH_REMATCH[1]}"
  peer_range_end="${BASH_REMATCH[2]}"

  if devices_overlap "${DEVICES}" "${PEER_devices}"; then
    die "NPU devices overlap active peer ${peer_label}: ${DEVICES} vs ${PEER_devices}"
  fi
  if [[ "${PORT}" == "${PEER_port}" ]]; then
    die "HTTP port ${PORT} is already used by active peer ${peer_label}"
  fi
  if [[ "${HCCL_MASTER_PORT}" == "${PEER_hccl_master}" ]]; then
    die "rendezvous port ${HCCL_MASTER_PORT} is already used by active peer ${peer_label}"
  fi
  if ranges_overlap \
    "${HCCL_RANGE_START}" "${HCCL_RANGE_END}" \
    "${peer_range_start}" "${peer_range_end}"; then
    die "HCCL socket ranges overlap active peer ${peer_label}"
  fi
  if port_in_range "${PORT}" "${peer_range_start}" "${peer_range_end}" ||
    port_in_range "${HCCL_MASTER_PORT}" "${peer_range_start}" "${peer_range_end}"; then
    die "HTTP or rendezvous port falls inside active peer ${peer_label} HCCL range"
  fi
  if port_in_range "${PEER_port}" "${HCCL_RANGE_START}" "${HCCL_RANGE_END}" ||
    port_in_range "${PEER_hccl_master}" "${HCCL_RANGE_START}" "${HCCL_RANGE_END}"; then
    die "active peer ${peer_label} ports fall inside this instance HCCL range"
  fi
  if paths_overlap "${RUN_ROOT}" "${peer_run}" ||
    paths_overlap "${LOG_ROOT}" "${peer_log}" ||
    paths_overlap "${RUN_ROOT}" "${peer_log}" ||
    paths_overlap "${LOG_ROOT}" "${peer_run}"; then
    die "run/log directories overlap active peer ${peer_label}"
  fi
}

is_healthy() {
  curl --noproxy '*' --silent --fail --max-time 2 \
    "http://${HEALTH_HOST}:${PORT}/system_stats" >/dev/null
}

validate_comfy_args() {
  local arg
  for arg in "${COMFY_ARGS[@]}"; do
    case "${arg}" in
      --listen|--port|--port=*|--database-url|--database-url=*|\
      --user-directory|--user-directory=*|--input-directory|\
      --input-directory=*|--output-directory|--output-directory=*|\
      --temp-directory|--temp-directory=*|--extra-model-paths-config|\
      --extra-model-paths-config=*|--disable-auto-launch|--cache-classic)
        die "${arg} is managed by run_comfy_instance.sh"
        ;;
    esac
  done
}

acquire_control_lock() {
  local waited=0
  local owner=""
  local lock_parent
  lock_parent="$(dirname "${CONTROL_LOCK_DIR}")"
  mkdir -p "${lock_parent}"
  while ! mkdir "${CONTROL_LOCK_DIR}" 2>/dev/null; do
    owner="$(cat "${CONTROL_LOCK_DIR}/owner.pid" 2>/dev/null || true)"
    if [[ "${owner}" =~ ^[1-9][0-9]*$ ]] && ! kill -0 "${owner}" 2>/dev/null; then
      rm -f "${CONTROL_LOCK_DIR}/owner.pid"
      rmdir "${CONTROL_LOCK_DIR}" 2>/dev/null || true
      continue
    fi
    if ((waited >= 10)); then
      die "another instance control operation holds ${CONTROL_LOCK_DIR}"
    fi
    sleep 1
    waited=$((waited + 1))
  done
  printf '%s\n' "$$" > "${CONTROL_LOCK_DIR}/owner.pid"
  CONTROL_LOCK_HELD=1
  trap 'release_control_lock' EXIT
}

release_control_lock() {
  if [[ "${CONTROL_LOCK_HELD:-0}" == "1" ]]; then
    rm -f "${CONTROL_LOCK_DIR}/owner.pid"
    rmdir "${CONTROL_LOCK_DIR}" 2>/dev/null || true
    CONTROL_LOCK_HELD=0
  fi
}

new_run_token() {
  local token
  token="$(od -An -N16 -tx1 /dev/urandom | tr -d '[:space:]')"
  if [[ ! "${token}" =~ ^[0-9a-f]{32}$ ]]; then
    die "failed to generate a run token"
  fi
  printf '%s\n' "${token}"
}

ensure_start_slot_available() {
  local pid_file="${RUN_ROOT}/comfyui.pid"
  local OLD_token OLD_pid OLD_devices OLD_port OLD_hccl_master
  local OLD_hccl_range OLD_run_root OLD_log_root OLD_state
  if [[ -f "${METADATA_FILE}" ]]; then
    read_metadata "${METADATA_FILE}" OLD
    if [[ "${OLD_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${OLD_pid}" 2>/dev/null; then
      if ! pid_has_instance_identity "${OLD_pid}" "${OLD_token}" "${INSTANCE_LABEL}"; then
        die "PID ${OLD_pid} is alive but does not prove ownership of instance ${INSTANCE_LABEL}; refusing to replace it"
      fi
      if ! requested_config_matches_metadata \
        "existing instance" "${OLD_devices}" "${OLD_port}" "${OLD_hccl_master}" \
        "${OLD_hccl_range}" "${OLD_run_root}" "${OLD_log_root}"; then
        die "existing instance ${INSTANCE_LABEL} is running with a different configuration"
      fi
      if is_healthy; then
        echo "instance ${INSTANCE_LABEL} already running: pid=${OLD_pid}, port=${PORT}"
        exit 0
      fi
      die "instance ${INSTANCE_LABEL} owns PID ${OLD_pid} but is not healthy"
    fi
    if [[ "${OLD_state}" != "stopped" ]]; then
      die "stale or unverified instance metadata at ${METADATA_FILE}; preserve it for inspection before starting"
    fi
  fi
  if [[ -e "${pid_file}" ]]; then
    die "stale PID file exists at ${pid_file}; refusing to delete or overwrite it"
  fi
}

launch_instance() {
  ensure_start_slot_available
  mkdir -p \
    "${RUN_ROOT}"/{user,input,output,temp,tmp,cache,ascend-log,workflows} \
    "${LOG_ROOT}"
  H3_RUN_TOKEN="$(new_run_token)"
  export H3_RUN_TOKEN
  write_metadata launching ""
  export \
    H3_INSTANCE_ID H3_INSTANCE \
    H3_NPU_DEVICES H3_PORT H3_HCCL_MASTER_PORT H3_HCCL_PORT_RANGE \
    H3_MASTER_ADDR H3_RUN_ROOT H3_LOG_ROOT H3_COMFY_HOME H3_PYTHON \
    H3_CANN_HOME H3_CANN_ENV H3_MODEL_CONFIG H3_EXTRA_ENV_FILE H3_LD_PRELOAD \
    H3_INSTANCE_METADATA H3_PEER_RUN_ROOT H3_PEER_LOG_ROOT \
    COMFYUI_MULTI_DEVICE_BACKEND
}

if [[ "${COMMAND}" != "status" && "${COMMAND}" != "logs" ]]; then
  if [[ "${COMMAND}" == "foreground" || "${COMMAND}" == "start" ]]; then
    validate_positive_integer H3_AVAILABLE_DEVICE_COUNT "${AVAILABLE_DEVICE_COUNT}"
    validate_positive_integer H3_EXPECTED_DEVICE_COUNT "${EXPECTED_DEVICE_COUNT}"
    validate_positive_integer H3_START_TIMEOUT "${START_TIMEOUT}"
    validate_positive_integer H3_STOP_TIMEOUT "${STOP_TIMEOUT}"
    if ((EXPECTED_DEVICE_COUNT > AVAILABLE_DEVICE_COUNT)); then
      die "H3_EXPECTED_DEVICE_COUNT exceeds H3_AVAILABLE_DEVICE_COUNT"
    fi
    validate_devices "${DEVICES}"
    validate_port H3_PORT "${PORT}"
    validate_port H3_HCCL_MASTER_PORT "${HCCL_MASTER_PORT}"
    validate_port_range H3_HCCL_PORT_RANGE "${HCCL_PORT_RANGE}"
    if [[ "${PORT}" == "${HCCL_MASTER_PORT}" ]]; then
      die "HTTP and rendezvous ports must differ"
    fi
    if port_in_range "${PORT}" "${HCCL_RANGE_START}" "${HCCL_RANGE_END}" ||
      port_in_range "${HCCL_MASTER_PORT}" "${HCCL_RANGE_START}" "${HCCL_RANGE_END}"; then
      die "HTTP and rendezvous ports must not fall inside this instance HCCL range"
    fi
    validate_comfy_args
  fi
  for pair in \
    "H3_RUN_ROOT=${RUN_ROOT}" \
    "H3_LOG_ROOT=${LOG_ROOT}" \
    "H3_PEER_RUN_ROOT=${PEER_RUN_ROOT}" \
    "H3_PEER_LOG_ROOT=${PEER_LOG_ROOT}" \
    "H3_COMFY_HOME=${COMFY_HOME}" \
    "H3_PYTHON=${PYTHON_BIN}" \
    "H3_CANN_HOME=${CANN_HOME}" \
    "H3_CANN_ENV=${CANN_ENV}" \
    "H3_MODEL_CONFIG=${MODEL_CONFIG}"; do
    validate_no_record_separators "${pair%%=*}" "${pair#*=}"
  done
  if [[ -n "${EXTRA_ENV_FILE}" ]]; then
    validate_no_record_separators H3_EXTRA_ENV_FILE "${EXTRA_ENV_FILE}"
  fi
  if [[ -n "${LD_PRELOAD_VALUE}" ]]; then
    validate_no_record_separators H3_LD_PRELOAD "${LD_PRELOAD_VALUE}"
  fi

  if [[ "${COMMAND}" == "start" || "${COMMAND}" == "foreground" ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "H3_PYTHON is not executable: ${PYTHON_BIN}"
    [[ -f "${COMFY_HOME}/main.py" ]] || die "missing ${COMFY_HOME}/main.py"
    [[ -f "${CANN_ENV}" ]] || die "missing CANN environment file: ${CANN_ENV}"
    [[ -f "${MODEL_CONFIG}" ]] || die "missing model config: ${MODEL_CONFIG}"
    if [[ -n "${EXTRA_ENV_FILE}" && ! -f "${EXTRA_ENV_FILE}" ]]; then
      die "missing H3_EXTRA_ENV_FILE: ${EXTRA_ENV_FILE}"
    fi
    acquire_control_lock
    check_peer_conflicts
  fi
fi

RUN_ROOT="$(normalize_path "${RUN_ROOT}")"
PEER_RUN_ROOT="$(normalize_path "${PEER_RUN_ROOT}")"
LOG_ROOT="$(normalize_path "${LOG_ROOT}")"
PEER_LOG_ROOT="$(normalize_path "${PEER_LOG_ROOT}")"
METADATA_FILE="${RUN_ROOT}/instance.json"

H3_INSTANCE_ID="${INSTANCE_ID}"
H3_INSTANCE="${INSTANCE_LABEL}"
H3_NPU_DEVICES="${DEVICES}"
H3_PORT="${PORT}"
H3_HCCL_MASTER_PORT="${HCCL_MASTER_PORT}"
H3_HCCL_PORT_RANGE="${HCCL_PORT_RANGE}"
H3_MASTER_ADDR="${MASTER_ADDR}"
H3_RUN_ROOT="${RUN_ROOT}"
H3_LOG_ROOT="${LOG_ROOT}"
H3_PEER_RUN_ROOT="${PEER_RUN_ROOT}"
H3_PEER_LOG_ROOT="${PEER_LOG_ROOT}"
H3_COMFY_HOME="${COMFY_HOME}"
H3_PYTHON="${PYTHON_BIN}"
H3_CANN_HOME="${CANN_HOME}"
H3_CANN_ENV="${CANN_ENV}"
H3_MODEL_CONFIG="${MODEL_CONFIG}"
H3_EXTRA_ENV_FILE="${EXTRA_ENV_FILE}"
H3_LD_PRELOAD="${LD_PRELOAD_VALUE}"
METADATA_FILE="${RUN_ROOT}/instance.json"
H3_INSTANCE_METADATA="${METADATA_FILE}"
PID_FILE="${RUN_ROOT}/comfyui.pid"
LOG_FILE="${LOG_ROOT}/comfyui.log"
export \
  H3_INSTANCE_ID H3_INSTANCE H3_NPU_DEVICES H3_PORT \
  H3_HCCL_MASTER_PORT H3_HCCL_PORT_RANGE H3_MASTER_ADDR \
  H3_RUN_ROOT H3_LOG_ROOT H3_PEER_RUN_ROOT H3_PEER_LOG_ROOT \
  H3_COMFY_HOME H3_PYTHON H3_CANN_HOME H3_CANN_ENV \
  H3_MODEL_CONFIG H3_EXTRA_ENV_FILE H3_LD_PRELOAD \
  H3_INSTANCE_METADATA
export H3_INSTANCE_METADATA

case "${COMMAND}" in
  start)
    launch_instance
    : > "${LOG_FILE}"
    nohup bash "${START_SCRIPT}" "${COMFY_ARGS[@]}" \
      </dev/null >>"${LOG_FILE}" 2>&1 &
    launched_pid=$!
    ready=0
    for _ in $(seq 1 "${START_TIMEOUT}"); do
      if is_healthy; then
        ready=1
        break
      fi
      if ! kill -0 "${launched_pid}" 2>/dev/null; then
        echo "instance ${INSTANCE_LABEL} exited during startup" >&2
        tail -n 160 "${LOG_FILE}" >&2 || true
        write_metadata failed ""
        exit 1
      fi
      sleep 1
    done
    if [[ "${ready}" != "1" ]]; then
      echo "instance ${INSTANCE_LABEL} was not healthy after ${START_TIMEOUT}s; process retained" >&2
      tail -n 160 "${LOG_FILE}" >&2 || true
      exit 1
    fi
    if [[ ! -f "${PID_FILE}" ]]; then
      echo "instance ${INSTANCE_LABEL} is healthy but has no PID metadata" >&2
      exit 1
    fi
    running_pid="$(tr -d '[:space:]' < "${PID_FILE}")"
    if [[ "${running_pid}" != "${launched_pid}" ]]; then
      echo "PID mismatch: launcher=${launched_pid}, pidfile=${running_pid}" >&2
      exit 1
    fi
    write_metadata running "${running_pid}"
    echo "instance ${INSTANCE_LABEL} ready: pid=${running_pid}, port=${PORT}, devices=${DEVICES}, backend=${MULTI_DEVICE_BACKEND}"
    ;;
  foreground)
    launch_instance
    release_control_lock
    trap - EXIT
    exec bash "${START_SCRIPT}" "${COMFY_ARGS[@]}"
    ;;
  status)
    CURRENT_token=""
    CURRENT_pid=""
    CURRENT_devices=""
    CURRENT_port=""
    CURRENT_state=""
    if [[ ! -f "${METADATA_FILE}" ]]; then
      echo "instance=${INSTANCE_LABEL} state=missing port=${PORT} run_root=${RUN_ROOT}"
      exit 1
    fi
    read_metadata "${METADATA_FILE}" CURRENT
    owned=no
    alive=no
    healthy=no
    if [[ "${CURRENT_pid}" =~ ^[1-9][0-9]*$ ]] &&
      kill -0 "${CURRENT_pid}" 2>/dev/null; then
      alive=yes
    fi
    if [[ "${alive}" == "yes" ]] &&
      pid_has_instance_identity "${CURRENT_pid}" "${CURRENT_token}" "${INSTANCE_LABEL}"; then
      owned=yes
    fi
    if is_healthy; then
      healthy=yes
    fi
    printf 'instance=%s state=%s pid=%s alive=%s owned=%s healthy=%s port=%s devices=%s backend=%s\n' \
      "${INSTANCE_LABEL}" "${CURRENT_state}" "${CURRENT_pid:-none}" \
      "${alive}" "${owned}" "${healthy}" "${CURRENT_port}" "${CURRENT_devices}" \
      "${CURRENT_multi_device_backend:-unknown}"
    if [[ "${owned}" == "yes" && "${healthy}" == "yes" ]]; then
      exit 0
    fi
    exit 1
    ;;
  logs)
    if [[ ! -f "${LOG_FILE}" ]]; then
      echo "log file does not exist: ${LOG_FILE}" >&2
      exit 1
    fi
    exec tail -n 200 -F "${LOG_FILE}"
    ;;
  stop)
    CURRENT_token=""
    CURRENT_pid=""
    acquire_control_lock
    if [[ ! -f "${METADATA_FILE}" ]]; then
      if [[ -e "${PID_FILE}" ]]; then
        die "PID file exists without ownership metadata; refusing to remove or signal it: ${PID_FILE}"
      fi
      echo "instance ${INSTANCE_LABEL} has no instance metadata; nothing to stop"
      exit 0
    fi
    read_metadata "${METADATA_FILE}" CURRENT
    if [[ ! "${CURRENT_pid}" =~ ^[1-9][0-9]*$ ]]; then
      echo "instance ${INSTANCE_LABEL} metadata has no PID; refused to modify stale state" >&2
      exit 1
    fi
    if ! kill -0 "${CURRENT_pid}" 2>/dev/null; then
      echo "instance ${INSTANCE_LABEL} PID ${CURRENT_pid} is not alive; stale metadata retained" >&2
      exit 1
    fi
    if ! pid_has_instance_identity "${CURRENT_pid}" "${CURRENT_token}" "${INSTANCE_LABEL}"; then
      echo "PID ${CURRENT_pid} does not prove instance ownership; refusing to signal it" >&2
      exit 1
    fi
    kill -TERM "${CURRENT_pid}"
    for _ in $(seq 1 "${STOP_TIMEOUT}"); do
      if ! kill -0 "${CURRENT_pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        H3_RUN_TOKEN="${CURRENT_token}"
        write_metadata stopped ""
        echo "instance ${INSTANCE_LABEL} stopped"
        exit 0
      fi
      sleep 1
    done
    echo "instance ${INSTANCE_LABEL} did not stop after SIGTERM; metadata and PID retained" >&2
    exit 1
    ;;
esac
