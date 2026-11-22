# R130 results — first native difference is Half GEMM with FP32 output

[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) remains open. The copy-stream
candidate in [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) is still disabled.
**Keep the diagnosis; do not adopt the deterministic setting without user approval.**

## 1. Original time-projection stages, not a replacement formula

[Original-stage diagnostic](time_stages130.py) loads only original FP32 time weights
and own-live NPU timesteps, using actual TP2 groups with USP4/world8. It calls the
unchanged R124 residual guard: four Half-operand GEMMs with FP32 output, three FP32
scaled additions, and the original optional local bias addition.

[Observer](projection_observer130.py) snapshots independent bytes immediately at
GEMM-return and before/after each inplace add, then returns the original result.
This avoids falsely reading GEMM0's later-mutated storage as its original output.
The observer is scoped to the private small test process/caller thread and restores
function identities on exit; it is **not** installed in the production worker.
Immediate D2H copies synchronize and may perturb timing/lifetime; these are not
clean performance runs. [CPU unit](observer_unit130.json) checks snapshot isolation,
return identity and exception restoration. It is not an NPU/model precision gate.

[Verified stages](verified_stages130.json) and [summary](stage_summary130b.json):

- All eight rank reports completed original → observed → fixed-operation → restored.
- The first changed stage on every rank is **MM0, Half × Half → FP32 output**, with
  all actual input operand bytes equal between repetitions. Maximum difference:
  **5.9604644775e-8**. The output shape is `[3, 2688]`.
- Isolated, fixed-input repetitions of all four GEMMs are non-bit-repeatable in
  **32/32 rank/operation groups**, up to **1.1920928955e-7** in a residual GEMM.
- All **28/28** fixed-input scaled-add/bias-add groups are bit-repeatable. Each add
  starts from the same frozen pre-add value, not an already-accumulated destination.
- Original FP32 parameter fingerprints are unchanged; process function objects are
  restored. Non-repeatability exists before the TP reduce and before later FP32
  additions. This does not rule out additional sources elsewhere in the full model.

Within-call stage operands are compared across their complete logical bytes. GEMM
outputs in the stage observer report different-byte counts, not per-element counts.
The fixed-operation and deterministic probes additionally compare every FP32 word.
Padding/physical-format bytes and internal CANN workspace are not fully audited.

## 2. Private deterministic capability probe

[Separate probe](deterministic_probe130.py) changes only its private process's standard
`torch.use_deterministic_algorithms` flag and restores the original false/warn-only
settings. No deprecated precision mode, SDK modification or production flag change.

It uses the original time-projection input and original cached high/low Half weights;
Half input decomposition follows the old guard formula. For each rank it repeats
MM0–MM3 and the full original local guard ten times under native false → true →
restored false. [Summary](determinism_summary130.json):

| Mode | Bit-repeatable rank/operation groups | Full-guard self-repeat max abs |
|---|---:|---:|
| Native, flag false | 0 / 40 | 5.9604644775e-8 |
| Deterministic, flag true | 40 / 40 | 0 |
| Restored, flag false | 0 / 40 | 5.9604644775e-8 |

This controlled reversal supports an original GEMM execution-path non-repeatability
finding. The exact internal CANN algorithm, split/reduction or workspace mechanism
has **not** been identified from a device-kernel trace; do not claim a specific
atomic-add implementation merely from the flag result.

### The deterministic output is stable but not original-bit-identical

For the full local time `proj_out` guard, deterministic versus the first native
output changes up to **2,084 / 8,064 FP32 words** on one rank:

- Maximum absolute difference: **5.9604644775e-8**.
- Maximum relative RMS error: **4.3729399142e-8** (ratio), or **0.00000437294%**.
- All four isolated GEMMs also differ from native; largest absolute difference:
  **1.1920928955e-7**.

These are component differences, **not** full-model video/audio error estimates.
The native R128/R129 final-output noise is not an authorized candidate tolerance.
No candidate flag/precision policy is adopted or inserted into model generation.
A full-50 diagnostic using the deterministic setting requires the user's decision.

### Timing scope is explicit

Maximum across ranks of **each rank's own median for repeats 1–9**, including
explicit synchronization and output D2H. This is **not** the median of per-repeat
rank-max timings used by a coordinated full-model benchmark:

| Operation | Native | Deterministic | Restored |
|---|---:|---:|---:|
| MM0 | 0.393 ms | 0.600 ms | 0.399 ms |
| Full local time guard | 1.621 ms | 2.284 ms | 1.537 ms |

These synchronized microcomponent timings are neither pure-kernel events nor clean
model latency; do not infer a full-step slowdown or speedup from them. Peak NPU
memory and pure device-kernel events were not collected, so no memory improvement
or kernel-name attribution is claimed. No 49-step request, encoder or VAE was run
here, so no model155 checks or output quality pass
is claimed for this round. Original 50-layer/FP16/FP32/TP contracts remain unchanged.

## 3. Failures and exact execution records

The first stage diagnostic incorrectly assumed every local projection had no bias
and expected only three additions. The original TP primary ranks 0/2/4/6 do have
local bias; ranks 1/3/5/7 do not. The diagnosis assertion failed, not the original
model math. [Exact initial script](executed_sources130a/time_stages130.py) and
[failed rank0 receipt](stages_rank0130.json) remain preserved.

The corrected [execution pins](launch_pins130b.json) include the actual optional
bias branch; they do not change the original guard or omit bias. That computation
succeeded. Its shell command subsequently exited 1 because the local SCP destination
directory did not exist. The directory was created and the **same** completed JSON
receipts were transferred, without rerunning compute or overwriting failed data.

[Exact successful stage source](executed_sources130b/time_stages130.py),
[exact deterministic probe source](executed_deterministic_probe130.py),
[original source preflight](preflight130.json), [cleanup](cleanup130.json) and per-rank
scalar/text evidence are retained. All private stage/probe processes exited; all
flags/function objects were restored. Small tests created no authentication IPC.
No driver/device reset, shared-process signals, shared SDK change, or model/activation
archive transfer to the client occurred.

## 4. Strict next gate — Awaiting user decision

The original native time GEMM path is demonstrably non-repeatable; a standard
private deterministic flag removes that local repetition noise but changes bits.
Do **not** close #15 or unblock #4 based on this component probe alone.

Recommended next experiment, only if authorized: bounded full-50 formal-worker
native → deterministic → restored diagnostics on the same own-live input, keeping
all 155 checks, original TP2/USP4 and FP32 protections. Measure time boundary, final
video/audio deltas, repeated outputs and rank-max scope; no 49-step generation, no
copy candidate, no adoption. The standard flag can affect other operators too, so
any full-graph result must explicitly state its broader process scope.

If authorization is skipped/declined, leave model flags false and keep copy work
blocked. Continue only independent original-path diagnosis or separately approved
work; never reinterpret component drift as a blanket precision allowance.
