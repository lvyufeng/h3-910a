#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${H3_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/envs/py312/bin/python}"
CANN_HOME="${ASCEND_HOME_PATH:-/home/mseco/Ascend/cann-9.2.0}"
PROBE="${ROOT}/scripts/hccl_allreduce_probe.py"
INSTANCE="${1:-a}"

case "${INSTANCE}" in
  a)
    DEVICES=0,1,2,3
    MASTER_PORT=29611
    SOCKET_RANGE=60000-60999
    ;;
  b)
    DEVICES=4,5,6,7
    MASTER_PORT=29612
    SOCKET_RANGE=61000-61999
    ;;
  *)
    echo "usage: $0 {a|b}" >&2
    exit 2
    ;;
esac

set +u
# shellcheck disable=SC1091
source "${CANN_HOME}/set_env.sh"
set -u

export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export HCCL_NPU_SOCKET_PORT_RANGE="${SOCKET_RANGE}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT}"
export PYTHONDONTWRITEBYTECODE=1

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=4 \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  "${PROBE}" --instance "${INSTANCE}"
