#!/bin/bash
# R194: does a larger Bc help? R193's fit says time = 83.5 ms + 2.922 us * tiles, so fewer
# tiles should be faster. This pushes Bc past 384 to find both the UB ceiling and the payoff.
# Measurement only: retiling changes accumulation order, so nothing here is a candidate.
set -uo pipefail
R=/home/mseco/Ascend/cann-9.2.0
source $R/set_env.sh >/dev/null 2>&1
D=/tmp/h3_fa194
rm -rf $D; mkdir -p $D/out; chmod 700 $D
cp /tmp/fa910a_sweep.py $D/; cp /tmp/sweep191.py $D/
export PYTHONPATH="$D:$PYTHONPATH"
export LD_LIBRARY_PATH=/tmp/h3_lib184:$R/opp/vendors/customize/op_api/lib:$R/lib64:${LD_LIBRARY_PATH:-}
export VLLM_OMNI_910A_FLASH_ATTN=1 VLLM_OMNI_910A_FP16_SEQ=1
unset VLLM_OMNI_910A_PERF

for cfg in "128:384" "128:512" "128:768" "128:1024" "128:1536"; do
  br=${cfg%%:*}; bc=${cfg##*:}
  echo "=== br_cap=$br bc_cap=$bc"
  H3_FA_BR_CAP=$br H3_FA_BC_CAP=$bc H3_FA_SWEEP_OUT=$D/out/cfg_${br}_${bc}.json \
    timeout 900 /home/mseco/vllm910a/bin/python -u $D/sweep191.py 2>&1 | tail -16
done

echo "=== SUMMARY (recomputing tiles on the padded N, which is what the kernel actually uses)"
python3 - <<'PY'
import json, glob, math
N, D, H = 43776, 128, 7
def pick_tile(n, cap, align=16):
    t = (cap // align) * align
    while t >= align:
        if n % t == 0: return t
        t -= align
    return None
rows = []
for f in sorted(glob.glob('/tmp/h3_fa194/out/cfg_*.json')):
    d = json.load(open(f))
    br, bc = pick_tile(N, d['br_cap']), pick_tile(N, d['bc_cap'])
    if br is None or bc is None:
        rows.append((d['br_cap'], d['bc_cap'], 'NO_TILE', None, d['median_ms'], None)); continue
    qb = N // br
    tiles = sum(math.ceil(((i+1)*br)/bc) for i in range(qb))
    flops = tiles * 4*br*bc*D*H
    rows.append((br, bc, tiles, flops/1e12, d['median_ms'],
                 flops/(d['median_ms']/1000)/1e12))
base_ms = 154.57   # R191 baseline at 128/304
print(f"{'br':>4} {'bc':>5} {'tiles':>7} {'TFLOP':>7} {'median_ms':>10} {'TFLOPS':>8} {'vs base':>8}")
for br, bc, tiles, tf, ms, rate in rows:
    sp = f"{base_ms/ms:.3f}x" if ms else '-'
    tfs = f"{tf:.3f}" if tf else '-'
    rt = f"{rate:.2f}" if rate else '-'
    print(f"{br:>4} {bc:>5} {str(tiles):>7} {tfs:>7} {ms:>10.2f} {rt:>8} {sp:>8}")
# linear check against R193's fit
print()
print("R193 fit: time = 83.53 + 2.922 us * tiles")
for br, bc, tiles, tf, ms, rate in rows:
    if ms and tiles:
        pred = 83.53 + 0.002922 * tiles
        print(f"  br={br} bc={bc}: pred {pred:.2f} ms vs meas {ms:.2f} ms -> {'OK' if abs(pred-ms)/ms < 0.08 else 'DEVIATES'}")
PY
