# R218 — exact padding-task trim: +8.1% local FA, −0.568 s measured DiT step

**One useful independent candidate was measured; nothing adopted or combined.**
All default sources, TP2+USP4, original 50 blocks, FP16 matrices/FP32 exceptions and 155 strict Half checks are unchanged. Live encoder TP8 configuration remains the contract; this round uses an **isolated DiT replay** with sealed conditioning, so it does not execute the live encoder, 49-step generation or final media. The ≤10 s/step objective is still open.

## 1. Current same-run baseline, not a sum of old measurements

One fresh 8-rank process replayed step25, with warmup1 and two measured repetitions per case:

| Case | Rank-max warm median (s) | Repetitions (s) |
|---|---:|---|
| Clean baseline | 17.997241 |17.944227,18.050255|
| Read-only instrumented forward |18.079715|18.077854,18.081575|
| Restored clean baseline |18.000089|18.028515,17.971663|

Instrumentation timing overhead is ~0.46%; restoration differs from the pre-baseline by ~0.003s. All eight ranks really use TP2 and USP4 with TP pairs/explicit SP groups in receipts. There are 155 strict checks per measured forward/rank and 50 actual FA ACL launches per instrumented forward.

Same-run instrumented scopes (max of per-rank medians over two repetitions):

- Native same-ACL-stream FA kernel events: **7.626564s/forward**, 50 calls/rank; wrapper7.757519s, padding0.084293s.
- Strict check subspans: qkv1.243991s, fc1 1.577556s, adaln0.065784s. These lie inside parent linear/region spans.
- USP pre/post:0.874747s/0.289418s, inclusive stream waits.
- Prefetch span:3.381338s, **queue/cross-stream-wait inclusive**, not pure transfer or step-critical H2D. Bytes are counted from actual pinned dtype-flat masters; every rank issues 50 prefetch calls.
- All parent/child and copy/compute spans overlap; **do not sum them** into a decomposition or subtract median children from another run's parents. The report is a current-run prioritization budget, not an exact critical-path trace.

[All-rank current-run summary](path_summary218.json) · [Source/timing profiler](path_profile218.py) · [Runner process pins](process218.json).

Native self-repeat is still not bitwise exact: full-prediction relative RMS2.554e-4(video),4.022e-4(audio). Instrumented/restored predictions vary too. This is reported, **not** a tolerance or evidence that instrumentation has been exonerated numerically. [Native/instrumented/restored metrics](selfrepeat_metrics218.json).

## 2. Rejected independent candidate: exact UB broadcast

The existing rowmax-vector broadcast passes through UB→L0C→UB→transpose. The candidate uses only `vec_trans_scatter` on sixteen identical input-row aliases, making (M,16) repeated-row output by bit copy. Original auto synchronization and all arithmetic remain.

- Standalone 2,048-word tests, including signedzero/subnormal/NaN payloads: baseline and candidate both exactly match expected words.
- Complete small FA output3,670,016 words: zero differences.
- Real FA output37,445,632 words: zero differences.
- Actual geometry B1H7 input41792→kernel43776, key_limit41769, Br128/Bc304/30cores/dense unchanged.
- Warm ACL kernel152.256→**153.152ms** (slower ~0.59%); restored152.238ms. Wrapper154.567→155.481ms.

**Reject: no speed gain.** This candidate was never combined with the trim experiment or installed in the model.

[Broadcast exact-word gate](broadcast_real_comparison218.json) · [Unit gate](broadcast_unit_candidate218.json) · [Source-only diff generator](generate_broadcast218.py).

## 3. Useful independent candidate: logical padding-task trimming

Original internal padding expands query/key storage41792→43776 to retain Br128/Bc304. The accepted wrapper masks invalid keys to exactly zero probability, then slices query output back to input length. The kernel nevertheless schedules fully invalid K blocks and fully discarded internal Q rows.

This candidate keeps **all GM storage shapes/strides, Br/Bc, original accumulation order/FP32 values, masks and automatic dependency synchronization**. After original initialization, only scheduling counts are changed:

```text
Tr = ceil(input_N / Br)       = 327   (baseline342)
Tc = ceil(valid_keys / Bc)    = 138   (baseline144)
last_Br = Br; last_Bc = Bc
```

Final partly-valid Q/K tiles retain their full original width and mask. All actual query rows and valid keys are scheduled. The work ratio327×138/(342×144)=0.916301: ~8.37% of padded tasks omitted. These are only internally-added/discarded query rows or completely invalid key blocks, **not sparse/causal/approximate attention or actual token removal**. Removing exact-zero accumulation still requires numerical checking for subtle floating-point behavior; it was not promoted purely by a proof sketch.

Candidate object SHA `052c49f2df2235bac2136b4630a05a71bd1b9b454d88c354bbbf5e254ab6e327`, actual symbol:

```text
trim218_q41792_b1_h7_n43776_d128_br128_bc304_c30_k41769_v3__kernel0
```

The cache key includes the query prefix and original key limit, so it cannot silently replace baseline binaries.

### Local actual-geometry measurement (seven warm repetitions)

| Variant | Wrapper median (ms) | Native ACL kernel (ms) |
|---|---:|---:|
| Baseline |154.597800|152.235001|
| Trim padding tasks |143.173370|140.852997|
| Restored baseline |154.582660|152.229004|

**Kernel1.080808×, −11.382ms/call.** Real output37,445,632 words all equal baseline; no nonfinite or saturated words; SHA `3b7bf78c6a9623303b83d32ccd0fd112e967c7f761bb5142354d6eebd010f21b` for all three runs. [Local word comparison](trim_comparison218.json) · [Candidate instance and raw time](trim_receipts/trim_real.json).

Boundary tests at N2048/Br128/Bc256 with valid-key lengths1792(exact boundary) and1791(partial boundary), removing one all-invalid K tile: each has1,835,008 output words with zero differences, no nonfinite or saturation. [Boundary raw gate](boundary_trim218.json). The earlier N2048/pad23 test retained all tiles; it was only a wrapper-control test, not proof of task-removal behavior. [Schedule coverage contracts](padding_plan_contracts218.json) are non-numerical checks, not substitution for gates.

## 4. Independently gated 8-rank model measurement

A second fresh 8-rank process tested only this FA candidate (no audit/GEMM/offload candidates combined). Its first gate forward calls **original and candidate on identical actual layer Q/K/V** and returns **original** results to the model. All400attention comparisons=50layers×8ranks, **14,978,252,800 FP16 elements**, zero differing words. Every gate-forward retains155strict checks; any rank difference would stop the candidate case and restore. The measured candidate only runs after global gate success.

Warmup1 plus2measured repetitions per case, step25:

| Variant | Rank-max median (s) | Two repetitions (s) |
|---|---:|---|
| Same-process pre-baseline |17.944454|17.940174,17.948734|
| **Only FA padding trim** |**17.376167**|17.384443,17.367891|
| Restored original |17.977355|17.969358,17.985352|

**Measured step saving0.568287s (3.168% time reduction), speedup1.032705×** versus pre-baseline. Gain versus restored0.601188s. This is actual isolated DiT forward wall timing, not50×local savings or combining it with historical R185 results. Real candidate launch counts are50gate,50warmup,100measured per rank; source/model/check count, allocation memory, input/kernel symbols and restored status are saved.

[8-rank single-candidate summary](trim_model_summary218.json) · [Rank0 gate/measurement receipt](trim_model_receipts/candidate_rank0.json) · [Fail-closed gate source](trim_stage218.py) · [Launch pins](process_trim218.json).

### Numerical gate limitation — not hidden

Whole-model fresh forwards are not bitwise equal. Native self-repeat vs native first has video RMS2.701e-4/audio3.989e-4; candidate repeats vs native first have:

| Prediction | Candidate repeat1 relRMS | Candidate repeat2 relRMS | Native self-repeat relRMS |
|---|---:|---:|---:|
| video |2.576684e-4|2.556050e-4|2.700971e-4|
| audio |3.988365e-4|4.442992e-4|3.988783e-4|

These are scalar differences on full predictions, not a user-authorized tolerance. Exact real-leaf Q/K/V comparisons are strong localized evidence, **not** full model/49-step proof. Candidate is disabled/restored; user must decide the per-item gate and adoption before any composition. No relaxation requested merely because the differences resemble native noise. [Full numeric metrics](trim_prediction_metrics218.json).

**State: Awaiting user decision/additional evidence. Default OFF.** Next useful checks: step0/48 actual-QKV gates and wider layout/mask edge tests, then user decision on this candidate; do not quietly add it to stageA/guard or claim10s.

## 5. Failures, restore and resource discipline

- First launch preflight iterated `.build.lock` as a kernel directory; assertion failed **before Popen or model launch**. Re-read receipt/process state and copied exactly one pinned baseline binary, then ran. No uncertain process replay.
- Unit test initially overwrote CANN's PYTHONPATH, producing `ModuleNotFoundError: tbe` before compute; restored the sourced path suffix and reran.
- Trim-model generator matched `event_profiler = None` twice; exact-match assertion stopped source generation and launch. Anchored the unique initialization context, AST-checked and relaunched.
- All private scripts have bounded execution budgets. Two model process groups completed exit0 and destroyed distributed state; no shared reset/process signal or SDK changes.
- Weight caches, original transformer/encoder/audit/FA SHA pins preserved. Only source/scalar JSON/text copied to the client; all prediction/FA tensors remained on ts-217 and were removed after scalar comparisons.
- All candidate binary caches/packages and host-only outputs removed, no lingering private model/FA process. Accepted default files unchanged. Owned ts-217 lease released; CUDA grants were released unused. No use of another session's dormant ts-216 lease.

[Cleanup](cleanup218.json) · [Preflight hypothesis receipt](start_issue218.json) · [FA Issue #5](https://github.com/lvyufeng/h3-910a/issues/5).
