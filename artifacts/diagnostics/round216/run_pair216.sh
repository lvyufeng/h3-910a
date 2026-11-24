#!/bin/bash
# R216 paired measurement; candidate paths never replace the production package.
set -uo pipefail
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
ROOT=/tmp/h3_fa216
export H3_FA_BR_CAP=128
export H3_FA_BC_CAP=304
export H3_FA_PROBE_DIR=$ROOT
export VLLM_OMNI_910A_PERF=1
for pair in "baseline /tmp/fa_pkg" "barrier $ROOT/pkg"; do
  read -r label pkg <<< "$pair"
  out="$ROOT/out/$label"
  mkdir -p "$out" "$ROOT/kernels/$label"
  export H3_FA215_PKG=$pkg
  export H3_FA215_KERNEL_DIR=$ROOT/kernels/$label
  export H3_FA_SWEEP_OUT=$out/result.json
  echo "BEGIN $label package=$pkg"
  /home/mseco/vllm910a/bin/python -u "$ROOT/sweep215.py" >"$out/log" 2>&1
  rc=$?
  echo "RC $label=$rc"
  tail -30 "$out/log"
done
