#!/usr/bin/env bash
# Compatibility entry point for the former control script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${H3_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONTROL="${ROOT}/scripts/run_comfy_instance.sh"

usage() {
  cat <<'USAGE'
Usage: runtime/h3-instance-control.sh {start|foreground|stop|status|logs} {a|b|1|2} [-- COMFY_ARGS...]
USAGE
}

COMMAND="${1:-status}"
case "${COMMAND}" in
  start|foreground|stop|status|logs) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
shift

INSTANCE="${1:-${H3_INSTANCE:-a}}"
case "${INSTANCE}" in
  a|A|1|b|B|2) ;;
  *)
    echo "error: instance must be a/b/1/2, got ${INSTANCE}" >&2
    exit 2
    ;;
esac
shift || true

[[ -f "${CONTROL}" ]] || {
  echo "error: missing control script: ${CONTROL}" >&2
  exit 2
}
exec bash "${CONTROL}" "${COMMAND}" "${INSTANCE}" "$@"
