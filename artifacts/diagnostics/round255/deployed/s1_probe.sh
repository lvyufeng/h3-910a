#!/bin/bash
# R255 S1: FA kernel pipe utilization profile + toolchain availability probe.
set -uo pipefail
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
export H3_FA215_LIB=/tmp/libfa910a.so
export H3_FA215_PKG=/tmp/fa_pkg
export H3_FA_PROBE_DIR=/tmp/h3_fa245b
export H3_FA215_KERNEL_DIR=/tmp/h3_fa245b/kernels/br128_bc304
export H3_FA_BR_CAP=128 H3_FA_BC_CAP=304 VLLM_OMNI_910A_PERF=1
export H3_FA_SWEEP_OUT=/tmp/h3_fa245b/out/msprof255.json
export TMPDIR=/tmp/msprof255_tmp ASCEND_WORK_PATH=/tmp/msprof255_cann
mkdir -p /tmp/msprof255_tmp /tmp/msprof255_cann /tmp/msprof255 /tmp/h3_fa245b/out
cd /tmp/h3_fa245b
timeout --signal=TERM --kill-after=30s 600 /home/mseco/Ascend/cann-9.2.0/bin/msprof \
  --output=/tmp/msprof255 --aic-metrics=PipeUtilization \
  --application="/home/mseco/vllm910a/bin/python -u /tmp/h3_fa245b/fa245_point.py" \
  > /tmp/msprof255.log 2>&1
echo "prof_rc=$?"
ls /tmp/msprof255/*/mindstudio_profiler_output/ 2>/dev/null | head -8

echo "=== toolchain: triton ==="
/home/mseco/vllm910a/bin/python -c "import triton, triton.__version__ as v; print('triton', v)" 2>&1 | tail -2
echo "=== toolchain: ascendc headers ==="
ls -d /home/mseco/Ascend/cann-9.2.0/*/include/ascendc 2>/dev/null | head -2
ls /home/mseco/Ascend/cann-9.2.0/include 2>/dev/null | head -8
echo "=== toolchain: msopgen / opc ==="
which msopgen 2>/dev/null; ls /home/mseco/Ascend/cann-9.2.0/bin/msopgen 2>/dev/null
echo "=== graph mode ==="
/home/mseco/vllm910a/bin/python -c "
import torch, torch_npu
print('npu_graph_attr', [a for a in dir(torch.npu) if 'graph' in a.lower()][:8])
print('compile_backends', torch.compiler.list_backends() if hasattr(torch.compiler,'list_backends') else 'n/a')
" 2>&1 | tail -4
