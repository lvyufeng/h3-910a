#!/bin/bash
# R210: run the HCCL determinism probe twice, with and without HCCL_DETERMINISTIC=1.
set -u
source /home/mseco/Ascend/cann-9.2.0/set_env.sh >/dev/null 2>&1
PY=/home/mseco/vllm910a/bin/python
D=/tmp/h3_hccl210
mkdir -p "$D"; chmod 700 "$D"
for MODE in unset 1; do
  echo "===== HCCL_DETERMINISTIC=$MODE ====="
  if [ "$MODE" = "1" ]; then export HCCL_DETERMINISTIC=1; else unset HCCL_DETERMINISTIC; fi
  timeout 600 torch_npu_run --standalone --nproc-per-node 8 "$D/hccl_determinism210.py" 2>&1 | \
    grep -E "shape|verdict|WROTE|Error|error" | tail -12
done
