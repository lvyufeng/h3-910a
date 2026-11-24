# R217 — dense-FA mode verified; scoped-sync fault; exact grouped-PV variants rejected

**Decision: Reject all R217 candidates. Default source, precision and execution paths unchanged.**
This is a single-card local operator experiment with fixed synthetic FP16 Q/K/V, not an 8-rank TP2+USP4 whole-step result. No model weights were loaded, downloaded or transferred.

## Real geometry and mode

The compiled baseline instance recorded:

- B=1, H=7, input_N=41792, kernel_N=43776, D=128; original_pad=23, valid keys=41769.
- Br=128, Bc=304, Tr=342, Tc=144, blockDim/core budget=30.
- `has_attn_mask=False`, `only_causal=False`, `prev_block_num=next_block_num=65536`.
- The strategy name `sparse` does **not** make this kernel causal. Every query tile traverses every key tile: 49,248 tiles/head, 344,736 over seven heads, including masked key tails.

[Baseline instance and timing](restored_final217.json) · [Scheduling source hash and actual fields](summary217.json).

At this compiled geometry, QK+PV perform approximately **6.868156 TFLOP/call** (padded dense work). The useful input/valid-key product is 6.256266 TFLOP/call. Native ACL events measure 152.18 ms, or **45.13 TFLOPS/card** for padded work. The historical 96.9 TFLOPS/card GEMM result is an empirical reference from another workload, **not** a physical ceiling. This ratio is not measured hardware occupancy/MFU and does not identify Cube, vector or memory stalls.

The old 24%/causal-FLOP interpretation is therefore withdrawn. No causal mask or sparse approximation was added in this experiment.

## Measurements — warm medians, seven repetitions each

| Actual variant | Wrapper wall (ms) | Native ACL kernel (ms) | Output words vs baseline | Decision |
|---|---:|---:|---:|---|
| Accepted baseline, pre-candidate | 154.571 | 152.180 | Reference | Keep baseline |
| P×V, K=128 grouping | 157.652 | 155.342 | 0 / 37,445,632 different | Reject: 2.08% slower kernel |
| P×V, K=128, thread_num=2 | 157.701 | 155.331 | 0 / 37,445,632 different | Reject: 2.07% slower kernel |
| Restored accepted baseline, after candidates | 154.510 | 152.168 | 0 / 37,445,632 different | Keep baseline restored |

The grouped single/double-buffer versions produce **identical object SHA** `48282047ae8a5cb3121b28715bb2b35f7a69f010ee2acdb72d9836328a69f2d0`. `thread_num=2` did not produce a distinct compiled arrangement here; timing noise is not a second optimization.

Baseline object SHA is `aec5f81addb2063d7be34731034916e1901e5ea42816202e648a2e85fbccc6c8`. Actual symbol:

```text
b1_h7_n43776_d128_br128_bc304_c30_k41769_v3__kernel0
```

The baseline and corrected candidate output SHA is `3b7bf78c6a9623303b83d32ccd0fd112e967c7f761bb5142354d6eebd010f21b`.

[All-word comparisons](exact_comparison217.json) explicitly compare every FP16 word; all corrected small and real-shape outputs plus restored baseline have zero differences, nonfinite values and saturated Half words. This does not prove all model inputs, rank states or 49 steps are equivalent. There was no combined/model adoption or request for a relaxed numerical gate.

[Grouped timing](real_pvgroup_raw217.json) · [Double-buffer timing](real_pvdouble_raw217.json) · [Final restore](restored_end217.json).

Cold compile/first-call time and peak allocator memory remain in each raw receipt. Real-shape baseline/candidates peak allocated memory is approximately 1.01 GiB; no model/offloaded weights are involved.

## Failed attempts — not omitted or reported as gains

1. **R217 source generator** initially matched a copied Chinese comment incorrectly. The exact-match assertion failed; no generated variant was uploaded. The shell lacked top-level fail-fast on that attempt, so it created the two private package directories but scp reported missing generated files. Fixed the literal and repeated with `set -euo pipefail` and AST checks.
2. **Scoped-sync candidate:** moved C_l0c back **inside** its original scope, invoked the same installed official MatMulImpl, disabled automatic sync and placed full PIPE_ALL barriers around the call/transfer. It compiled, but actual small-shape execution failed with `507015` and L0A/L0B read/write conflicts. External barriers do not establish the generator's internal MTE1↔Cube dependencies. Candidate immediately stopped, no valid output/timing, no device reset. A fresh baseline process then executed the real shape successfully. [Full failure receipt](scoped_failure_raw217.json).
3. **Direct official P×V candidate:** automatic synchronization and FP32 output remained, but the official matmul input layout differs from V's original N1,K,N0 orientation. Small-shape output differed in 3,669,614/3,670,016 words (99.989%), relative RMS=1.41094335, max absolute difference=0.36193848. This is an implementation error, **not** acceptable rounding. Reject, no real-shape promotion. [Numeric failure](comparison_pv217.json).
4. **Corrected grouped P×V:** retained original transposed V loads and automatic synchronization. It passed the above exact local comparisons but was slower; rejected on performance.
5. **Torch event timing bug in this probe:** queued torch events around a direct ctypes ACL launch measured ~0.005 ms for six of seven long kernels while wall time remained 154 ms. Those kernel events are invalid. Raw receipts are retained. Final kernel timing uses native `aclrtCreateEventWithFlag(..., ACL_EVENT_TIME_LINE)` / `aclrtRecordEvent` directly on the **same ACL stream**, with prebuilt NZ inputs, return-code checks and end-event synchronization; repeated values are approximately 152.17–152.18 ms. Clean wrapper-wall measurement runs separately. [Corrected probe](probe217.py).

The original R216 sharing generator is still a failed draft, not a successfully generated/compiled implementation. [R216 failure inventory](../round216/RESULTS.md). R215 tile variants have smaller gains but lack paired exact-output proof; see [R215 qualifications](../round215/RESULTS.md).

## 10 s/step implication — scenarios, not measured gains

No new whole-step speedup was achieved. The paired R185 audit candidate remains a different experiment: **17.987→15.473 s**, saving 2.514 s, not adopted by this round. Original guards and all 155 checks remain enabled by default.

A conditional scenario using different-era scopes is:

```text
new_step = A - guard_gain - F + F / FA_speedup - other_gain
A = 15.473 s; F = 7.454 s (historical FA scope)
```

With an **unproved** 1.665 s guard gain and no other gain, FA must improve by approximately **2.044×** to reach 10 s. FA 1.5× gives 11.323 s; 1.89× gives 10.298 s. Without that assumed guard gain, the fixed-other budget would require 3.763×. None is a physical lower bound, a necessity proof for removing guards, or evidence that a candidate can achieve that speed.

R193's linear fit does not prove its intercept or slope is removable synchronization cost. The dense-mode finding makes previous causal-normalized feasibility claims unreliable. Next: measure a current same-run critical-path budget, then investigate exact vector/layout memory passes and correct irregular-tail tiling rather than blindly bypassing automatic dependencies. The optimization objective stays active; encoder TP8/A2-A3 remain open, video VAE remains user-paused.

## Cleanup and provenance

- Original `/tmp/fa_pkg` source hashes unchanged: fwd `a21c52e…`, utils `6aa8949d…`.
- No shared SDK/driver changes, device reset or unrelated process signaling.
- All synthetic output archives removed **on ts-217** after scalar comparisons; none transferred to client.
- Private candidate packages and candidate cache/temporary trees removed; scalar receipts and accepted baseline source retained.
- Owned ts-217 lease released (`released=true`); CUDA grants were released without use.
- Installed proprietary SDK snapshots are kept locally read-only and excluded from version control; publish only hashes/API signatures and our diagnostic code.

[Cleanup receipt](cleanup217.json) · [Replay summary generator](summarize217.py) · [Issue #5](https://github.com/lvyufeng/h3-910a/issues/5).
