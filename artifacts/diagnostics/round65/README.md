# Round65 — exact FP32 conversion-cache ablation in the stable text stack

## Measured result

The combined cache improves this resident experimental forward from **4.19700s
to3.63699s (1.154×)**, with all full outputs raw-bit identical. It **does not
reach2s**, and costs **11.1084GiB of extra FP32 weights per rank**. No production
adoption is made; this experiment quantifies rather than assumes the remaining
cast overhead.

Warm times are medians of three per-call rank maxima:

- Fixed baseline:4.19700s (4.18532–4.22465s).
- Activation-only reuse:4.03619s (1.040×), no new weight shadow.
- FP32 weight shadow only:3.82297s (1.098×).
- Both:3.63699s (3.62892–3.65273s,1.154×).
- Original math/dispatch restored:4.22081s (+0.567% drift; shadows still resident).

[Verified summary](<summary.json>) and [rank0 raw receipt](<ranks/shadow_rank0.json>)
retain all eight ranks, first/check-bearing calls and clean warm times. All
**160 complete outputs** (5 modes×4 forwards×8 ranks), each10,460,160 values,
match the original baseline; that baseline also reproduces the previous Round60
complete text output on every rank. Additionally6,400 same-input projection
checks cover the Q/K/V/merged gate-up outputs and the FP32 O/down partials.
No final-output subsampling or RMS-only acceptance is used.

HBM snapshots rise from8,280,360,448 allocated /8,921,284,608 reserved bytes/rank
to20,208,014,848 allocated /21,923,627,008 reserved bytes/rank. These are actual
post-call snapshots, **not peak HBM measurements**. They exclude a full serving
pipeline and cannot prove that its other components coexist with these shadows.

## Scope before speed

This is an **isolated resident experimental50-layer text-only TP8** A/B, not the
production text-encode stage. It reuses the actual captured2043-token inputs and
real checkpoint weights, the previous experimental FP32 Linear/causal math,
1024-output K/V padding, HF-math Norm, native BF16-GM SwiGLU and native-round TBE
RoPE. Processor, independent vision encoding, embedding vocabulary lookup,
model loading/offload and full H3 serving are outside the warm forward timer.
No production source or checkpoint is modified. Full-service text≤2s, original
BF16/HF-oracle equivalence and official lossless generation are not claimed.

## Controlled changes

[Cache helper](<shadow_cache65.py>) has two independent mechanisms:

1. **Weight shadow:** cache the exact result of the original `weight.float()` for
   Q, O, merged gate/up and down projections.200 entries across50 layers retain
   11,927,552,000 FP32 bytes/rank (11.1084GiB). The original BF16 weights stay intact.
   K/V already have100 padded FP32 weights (2,097,152,000 bytes/rank) in the existing
   [padding helper](<source/padded_fp32_projection59.py>); these are not duplicated.
2. **QKV activation reuse:** one `hidden_states.float()` lives only inside one
   attention forward. The original BF16 input and three separate Q/K/V GEMMs stay
   intact, as do padded K/V GEMM→crop128→contiguous→BF16. There is no global
   pointer-keyed activation cache or cross-forward reuse.

The [reproducer](<text_shadow_tp8.py>) keeps the previous **merged** gate/up GEMM,
not the split gate/up code in the unmodified candidate class. O/down preserve
FP32 partial GEMM→the returned FP32 all-reduce tensor→BF16 output exactly. Norm,
causal attention, RoPE and all collective ordering are fixed across cases.

Five modes run in one model/process group: baseline, activation-only,
weight-shadow, both (`shared_qkv`), original math restored. The last mode keeps
already allocated shadows resident, so it is **math/dispatch restoration only**,
not restoration of the baseline HBM footprint. All shadows are cleared during
final cleanup. Model/DiT/VAE co-residency is not inferred from a text-only HBM fit.

## Identity and timing gates

- Baseline must reproduce the previous Round60 **complete final text tensor**
  on every rank before any new candidate is accepted. It is not replaced with
  an independently invented oracle.
- First candidate forwards compare same-input Q/K/V and merged gate/up outputs:
  200 checks/rank. Weight/shared modes additionally compare100 FP32 O/down
  partials **before** all-reduce, totaling300 checks/rank. Actual final tensors
  include the original all-reduce/cast/deepstack path.
- All modes compare the full final output to baseline, including first plus3
  warm forwards on all8 ranks. Input embedding/positions/mask/DeepStack and **all
  actual model parameter bytes** are hashed before and after the entire run.
- Reference checks, CPU downloads, tensor hashing, cache population and kernel
  first use are outside clean warm measurements. A start barrier+device drain
  precedes each timer; the end explicitly drains outstanding device work.
- The report takes `median(max_rank(same warm call))`, never
  `max_rank(median(warm calls))`. First calls are separately retained and include
  correctness comparisons/cache construction; they are not warm performance.

The [aggregator](<summarize.py>) rejects missing ranks, incomplete runs, source
mismatches, changed input/weight bytes, wrong baseline, changed final tensor bits,
missing same-input checks, unexpected padding/shadow footprints and leaked
activation scopes. It reports allocated/reserved HBM snapshots, **not an unmeasured
peak HBM**, and does not hide the extra shadow storage.

## Cache lifecycle is not a general loader patch

Entries include source version, data/storage pointer, storage byte size, offset,
shape, stride, dtype and device, and hold source references to prevent freed
storage reuse. Unreadable inference-tensor weight versions and grad-enabled
execution bypass weight caching. Entry/byte caps fail rather than silently thrash.

**`.data.copy_` can bypass Parameter version counters.** Actual existing loaders
use `.data` writes; a version-only cache is not a general reload-safe solution.
This diagnostic constructs shadows only after loading and freezes/byte-hashes the
model. Every loader update, `.data` write, offload or external state replacement
requires explicit `clear()` before reuse. The portable tests deliberately expose
that limitation, not claim invisible `.data` mutations are automatically detected.
Strong source references can retain old HBM until cleared. Cache hit/miss and
activation conversion/hit counters in the raw receipts are **cumulative across
modes**, not per-call stage costs. The actual new weight cache stays at200 entries;
no eviction/recompilation policy is changed to inflate a warm result.

[Actual-NPU lifecycle controls](<cache_guard_npu.json>) separately pass13 checks:
ordinary in-place version invalidation, exact conversion values, storage rebinding,
fresh same-storage views, offset/stride changes, inference-weight bypass, activation
scope release and clear/reset. The NPU test explicitly observes that `.data.add_`
does **not** change the Parameter version; correctness after that operation requires
`clear()`. These controls do not pretend a diagnostic cache is an integrated
loader/offload solution.

Intermediate activations created inside inference mode have no version counter.
Reuse for those is restricted to the same live object in the inspected read-only
attention call. It is not a guarantee against arbitrary external unversioned
in-place writes. The actual same-input projections and full-run state checks,
not a mock arithmetic emulator, certify this workload.

## Reproduction

```bash
python3 -m unittest discover -s artifacts/diagnostics/round65 -p 'test_*.py' -v
python3 artifacts/diagnostics/round65/summarize.py
```

NPU runs require a fresh owned lease, the retained checkpoint/fixtures and CANN
setup on ts-217. The existing run ID and evidence claims must never be overwritten
or replayed while active. Source snapshots retain the exact previous experiment
bytes. [Final source verification](<final_sources.json>) matches23 production/
model/dedicated-module fingerprints and12 diagnostic/prior-kernel dependencies.
All managed jobs have finished. [Live-probed cleanup](<cleanup_receipt.json>) records
eight cards at0% utilization, zero accelerator processes and the lease released.
[Portable regression receipt](<test_receipt.json>) records17 new lifecycle/identity/
timing tests plus41 prior service/kernel/block tests, all58 passing; the13 actual
NPU lifecycle controls are recorded separately and not counted as portable tests.

## Next gate and remaining objective

This removes only~0.56s of a4.20s resident experimental forward; it is evidence
against expecting weight/activation cast caching alone to reach2s. The next
useful measurement is a synchronized profiler of this exact, now bit-stable stack,
particularly the remaining unfused **native FP32 Norm stats / BF16 rounding/gamma**
path and FP32 projection/collective balance. Do not reuse old native-GQA70s
attribution for this3.64s variant, or adopt the prior faster Norm whose full-stack
identity already failed. Any exact split/fused Norm candidate must preserve
residual and normalized-BF16 rounding boundaries and pass same-input plus full
stack checks before speed is counted.

Text service≤2s, native audio repeatability/full-A/V acceptance, strict complete
DiT-step<15s margin,49-step FL2VA quality and official task/precision/warm-protocol
matched A2/A3 comparison remain open. The overall goal stays active.
