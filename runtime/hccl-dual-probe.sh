#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${H3_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${H3_LOG_ROOT:-${ROOT}/logs}/hccl-dual"
mkdir -p "${LOG_DIR}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_a="${LOG_DIR}/group-a-${stamp}.log"
log_b="${LOG_DIR}/group-b-${stamp}.log"

bash "${SCRIPT_DIR}/hccl-group-probe.sh" a >"${log_a}" 2>&1 &
pid_a=$!
bash "${SCRIPT_DIR}/hccl-group-probe.sh" b >"${log_b}" 2>&1 &
pid_b=$!

status_a=0
status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?

echo "group A log: ${log_a}"
echo "group B log: ${log_b}"
if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
  echo "HCCL dual probe failed: A=${status_a}, B=${status_b}" >&2
  tail -120 "${log_a}" >&2 || true
  tail -120 "${log_b}" >&2 || true
  exit 1
fi

grep -h "HCCL_PROBE" "${log_a}" "${log_b}" | sort
