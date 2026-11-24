#!/bin/bash
# One bounded, private FA process. No destructive cleanup or global environment edits.
set -euo pipefail
label=${1:?label required}
variant=${2:?variant required}
n=${3:?sequence length required}
case "$label" in small_base|small_scoped|small_pv|small_pvgroup|small_pvdouble|real_base|real_scoped|real_pv|real_pvgroup|real_pvdouble|real_restored|real_restored_final|real_restored_end) ;; *) exit 2;; esac
case "$variant" in baseline|scoped|pv|pvgroup|pvdouble) ;; *) exit 2;; esac
root=/tmp/h3_fa217
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export H3_FA215_KERNEL_DIR=$root/kernels/$label
export H3_FA215_PKG=$root/${variant}_pkg
export H3_FA_BR_CAP=128 H3_FA_BC_CAP=304
export VLLM_OMNI_910A_PERF=1
export TMPDIR=$root/tmp/$label
export ASCEND_WORK_PATH=$root/cann/$label
mkdir -p "$H3_FA215_KERNEL_DIR" "$TMPDIR" "$ASCEND_WORK_PATH" "$root/logs"
chmod 700 "$H3_FA215_KERNEL_DIR" "$TMPDIR" "$ASCEND_WORK_PATH"
cd "$root"
set +e
timeout --signal=TERM --kill-after=20s 480 /home/mseco/vllm910a/bin/python -u "$root/probe217.py" --label "$label" --n "$n" --pad 23 >"$root/logs/$label.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$root/logs/$label.rc"
printf 'R217 label=%s variant=%s exit=%s\n' "$label" "$variant" "$rc"
/home/mseco/vllm910a/bin/python - "$root/out/$label/result.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print('NO_RESULT'); raise SystemExit(0)
d=json.loads(p.read_text())
print(json.dumps({k:d[k] for k in ['label','complete','error','warm_wrapper_wall_median_ms','kernel_event_median_ms','output_sha256','compiled_kernels'] if k in d}))
PY
exit "$rc"
