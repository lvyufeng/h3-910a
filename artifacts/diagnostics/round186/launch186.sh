#!/bin/bash
set -euo pipefail
R=/home/mseco/Ascend/cann-9.2.0
D=/tmp/h3_verify186
LABEL=verify186
source $R/set_env.sh
# prepend the audit and model trees but keep set_env.sh's own PYTHONPATH: dropping it
# removes the CANN python packages and the run dies with "No module named tbe"
export PYTHONPATH="$D/audit:$D:/tmp/h3_align12401:${PYTHONPATH:-}"
export H3_VERIFY186_LABEL=$LABEL
export H3_VERIFY186_STEPS=${H3_VERIFY186_STEPS:-10}
export H3_FAST_AUDIT=verify
export H3_FAST_VERIFY_OUT=$D/out/verify186
export H3_FAST_LIB=/tmp/h3_lib184/libh3fast184.so
export LD_LIBRARY_PATH=$R/opp/vendors/customize/op_api/lib:$R/lib64:${LD_LIBRARY_PATH:-}
export ASCEND_WORK_PATH=$D/cann_$LABEL
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=7200
export VLLM_OMNI_910A_FLASH_ATTN=1 VLLM_OMNI_910A_FP16_SEQ=1 VLLM_OMNI_910A_TE_DTYPE=fp16
export VLLM_OMNI_910A_TEXT_HF_MATH=1 VLLM_OMNI_910A_NO_FUSED_ACT=1 VLLM_OMNI_910A_NO_FUSED_NORM=1
export H3_NPU_NAN_COMPAT124=1 H3_NPU_VAE_PRESERVE_FP32_PARAMS124=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
mkdir -p $D/$LABEL
cd $D
/home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 $D/run186.py \
   > $D/out/verify186.log 2>&1
echo "exit=$?"
tail -4 $D/out/verify186.log
echo "=== verify receipts"
ls -l $D/out/verify186.rank*.json 2>/dev/null | head -10
python3 -c "
import json,glob
tot=mis=0; hits=0; sat=0
for f in sorted(glob.glob('$D/out/verify186.rank*.json')):
    d=json.load(open(f))['summary']
    tot+=d['comparisons']; mis+=d['mismatches']; hits+=d['nonfinite_hits']; sat+=d['saturated_hits']
    print(f\"  rank{d['rank']}: comparisons={d['comparisons']} mismatches={d['mismatches']} agree={d['agree']}\")
print(f'TOTAL comparisons={tot} mismatches={mis} nonfinite_hits={hits} saturated_hits={sat}')
print('VERDICT:', 'ALL AGREE' if tot>0 and mis==0 else 'MISMATCH FOUND' if mis else 'NO DATA')
"
