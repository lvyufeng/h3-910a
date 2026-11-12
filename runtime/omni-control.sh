#!/usr/bin/env bash
# Start/stop the vLLM-Omni compatible H3 service that fronts one ComfyUI instance.
#
# Usage:
#   bash runtime/omni-control.sh start [smoke|target]
#   bash runtime/omni-control.sh stop
#   bash runtime/omni-control.sh status
#
# Environment: COMFY_URL, OMNI_PORT, OMNI_WORKFLOW, INSTANCE_TAG
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PROFILE="${2:-smoke}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8801}"
OMNI_PORT="${OMNI_PORT:-9101}"
WORKFLOW="${OMNI_WORKFLOW:-artifacts/design/workflows/instance-1-${PROFILE}.json}"
PIDFILE="${PIDFILE:-run/omni/omni-${OMNI_PORT}.pid}"
LOGFILE="${LOGFILE:-logs/omni/server-${OMNI_PORT}.log}"
PYTHON="${PYTHON:-envs/py312/bin/python}"

mkdir -p run/omni logs/omni

running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE")"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

case "${1:-status}" in
  start)
    if running; then
      echo "already running: pid $(cat "$PIDFILE")"
      exit 0
    fi
    [[ -f "$WORKFLOW" ]] || { echo "workflow not found: $WORKFLOW" >&2; exit 1; }
    echo "starting omni service on :$OMNI_PORT -> $COMFY_URL ($WORKFLOW)"
    setsid nohup "$PYTHON" scripts/h3_vllm_omni_server.py \
      --comfy "$COMFY_URL" \
      --workflow "$WORKFLOW" \
      --port "$OMNI_PORT" \
      --output-dir run/omni \
      ${OMNI_ARGS:-} \
      >> "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    disown || true
    sleep 2
    if running; then
      echo "started: pid $(cat "$PIDFILE")"
    else
      echo "failed to start; see $LOGFILE" >&2
      tail -20 "$LOGFILE" >&2 || true
      exit 1
    fi
    ;;
  stop)
    if running; then
      pid="$(cat "$PIDFILE")"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
      echo "stopped: pid $pid"
    else
      echo "not running"
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if running; then
      echo "running: pid $(cat "$PIDFILE") port $OMNI_PORT"
      curl -s --noproxy '*' "http://127.0.0.1:${OMNI_PORT}/health" || true
      echo
    else
      echo "not running"
      exit 1
    fi
    ;;
  *)
    echo "usage: $0 {start [smoke|target]|stop|status}" >&2
    exit 2
    ;;
esac