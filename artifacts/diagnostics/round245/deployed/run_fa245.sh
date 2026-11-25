#!/bin/bash
# R245: standalone FA tile sweep with paired numeric drift. Measurement only.
set -uo pipefail
root=/tmp/h3_fa245b
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export H3_FA215_LIB=/tmp/libfa910a.so
export H3_FA215_PKG=/tmp/fa_pkg
export H3_FA_PROBE_DIR=$root/probe
export VLLM_OMNI_910A_PERF=1
mkdir -p "$root/out" "$root/kernels" "$root/logs" "$root/tmp" "$root/cann"
mkdir -p "$root/probe"
cp "$root"/fa910a_sweep.py "$root/probe"/fa910a_sweep.py
ref="$root/out/baseline_128_304.npy"
configs="128:304 64:768 64:912 64:608 64:456 96:608 96:456 128:608 128:456 128:768"
for cfg in $configs; do
  br=${cfg%%:*}; bc=${cfg##*:}
  label="br${br}_bc${bc}"
  export H3_FA_BR_CAP=$br H3_FA_BC_CAP=$bc
  export H3_FA215_KERNEL_DIR=$root/kernels/$label
  export H3_FA_SWEEP_OUT=$root/out/$label.json
  export TMPDIR=$root/tmp/$label
  export ASCEND_WORK_PATH=$root/cann/$label
  mkdir -p "$H3_FA215_KERNEL_DIR" "$TMPDIR" "$ASCEND_WORK_PATH"
  if [ "$label" = "br128_bc304" ]; then
    export H3_FA_SAVE_NPY=$ref
    unset H3_FA_REF_NPY
  else
    unset H3_FA_SAVE_NPY
    export H3_FA_REF_NPY=$ref
  fi
  echo "=== $label ==="
  timeout --signal=TERM --kill-after=20s 300 /home/mseco/vllm910a/bin/python -u "$root/fa245_point.py" \
    >>"$root/logs/$label.log" 2>&1
  echo "rc=$? $label"
  grep -E '^\{"br"' "$root/logs/$label.log" | tail -1 || tail -2 "$root/logs/$label.log"
done
echo DONE_R245
