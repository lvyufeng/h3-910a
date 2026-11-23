#!/bin/bash
# R191: sweep bc_cap (and one br_cap control) on the real FA shape, one process per config
set -uo pipefail
R=/home/mseco/Ascend/cann-9.2.0
source $R/set_env.sh >/dev/null 2>&1
D=/tmp/h3_fa191
mkdir -p $D/out; chmod 700 $D
cp /tmp/fa910a_sweep.py $D/
cp /tmp/sweep191.py $D/
export PYTHONPATH="$D:$PYTHONPATH"
export LD_LIBRARY_PATH=/tmp/h3_lib184:$R/opp/vendors/customize/op_api/lib:$R/lib64:${LD_LIBRARY_PATH:-}
export VLLM_OMNI_910A_FLASH_ATTN=1 VLLM_OMNI_910A_FP16_SEQ=1
unset VLLM_OMNI_910A_PERF

for cfg in "128:304" "128:256" "128:384" "128:128" "64:304"; do
  br=${cfg%%:*}; bc=${cfg##*:}
  echo "=== br_cap=$br bc_cap=$bc"
  H3_FA_BR_CAP=$br H3_FA_BC_CAP=$bc H3_FA_SWEEP_OUT=$D/out/cfg_${br}_${bc}.json \
    timeout 900 /home/mseco/vllm910a/bin/python -u $D/sweep191.py 2>&1 | tail -14
done

echo "=== SUMMARY"
python3 - <<'PY'
import json, glob
rows = []
for f in sorted(glob.glob('/tmp/h3_fa191/out/cfg_*.json')):
    try:
        rows.append(json.load(open(f)))
    except Exception as e:
        print('bad', f, e)
if not rows:
    print('NO DATA'); raise SystemExit
base = next((r for r in rows if r['br_cap'] == 128 and r['bc_cap'] == 304), None)
print(f"{'br':>4} {'bc':>4} {'tiles':>7} {'median_ms':>10} {'TFLOPS':>8} {'vs base':>8} finite")
for r in rows:
    speed = f"{base['median_ms']/r['median_ms']:.3f}x" if base else '-'
    print(f"{r['br']:>4} {r['bc']:>4} {r['tiles']:>7} {r['median_ms']:>10.2f} "
          f"{r['achieved_TFLOPS_per_card']:>8.2f} {speed:>8} {r['output_finite']}")
PY
