#!/bin/bash
set -euo pipefail
label=${1:?label required}
n=${2:?sequence required}
trim=${3:?trim flag required}
case "$label" in base_small|trim_small|base_real|trim_real|restored_real) ;; *) exit 2;; esac
case "$trim" in 0|1) ;; *) exit 2;; esac
root=/tmp/h3_trim218
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
export H3_FA215_KERNEL_DIR=$root/kernels/$label H3_FA215_PKG=$root/pkg
export H3_FA_BR_CAP=128 H3_FA_BC_CAP=304
export VLLM_OMNI_910A_PERF=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=$root/tmp/$label ASCEND_WORK_PATH=$root/cann/$label
if [ "$trim" = 1 ];then export H3_FA_TRIM_Q_LIMIT="$n";else unset H3_FA_TRIM_Q_LIMIT;fi
mkdir -p "$H3_FA215_KERNEL_DIR" "$TMPDIR" "$ASCEND_WORK_PATH" "$root/logs"
cd "$root"
set +e
timeout --signal=TERM --kill-after=20s 480 /home/mseco/vllm910a/bin/python -u "$root/trim_probe218.py" --label "$label" --n "$n" --pad 23 >"$root/logs/$label.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" >"$root/logs/$label.rc"
printf 'TRIM label=%s exit=%s\n' "$label" "$rc"
/home/mseco/vllm910a/bin/python - "$root/out/$label/result.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.exists():
 d=json.loads(p.read_text())
 print(json.dumps({k:d[k] for k in ['label','complete','error','warm_wrapper_wall_median_ms','kernel_event_median_ms','output_sha256','compiled_kernels'] if k in d}))
else:print('NO_RESULT')
PY
exit "$rc"
