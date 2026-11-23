#!/bin/bash
set -uo pipefail
R=/home/mseco/Ascend/cann-9.2.0
source $R/set_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=$R/aarch64-linux/devlib:$R/aarch64-linux/devlib/linux/aarch64:$R/lib64:${LD_LIBRARY_PATH:-}
D=/tmp/h3_prof192
rm -rf $D; mkdir -p $D/out; chmod 700 $D
cp /tmp/fa_app192.py $D/; cp /tmp/fa910a_sweep.py $D/
export PYTHONPATH="$D:/tmp/h3_fa191:$PYTHONPATH"
unset VLLM_OMNI_910A_PERF

echo "=== version"
$R/tools/msopprof/bin/msopprof -V 2>&1 | head -3

echo "=== profile"
cd $D
# msopprof takes the application as a positional argument; there is no "--" separator
H3_FA_CALLS=6 $R/tools/msopprof/bin/msopprof \
  --aic-metrics PipeUtilization,ArithmeticUtilization \
  --output $D/out \
  /home/mseco/vllm910a/bin/python $D/fa_app192.py 2>&1 | tail -16

echo "=== csv files"
find $D/out -name "*.csv" 2>/dev/null | head -12
