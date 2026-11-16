# Round71 — exact vision Linear activation/shadow factors on complete encode_ids

## Main result: small exact gain, not complete encoder2s

With the proven fast text stack fixed, five independent vision Linear modes
measure the complete resident **encode_ids** stage. All three warmed same-call
rank maxima are included per mode; first checks/compiler/cache calls are excluded.

- Native vision conversions: **6.92700s** median.
- Integer-codec activation expansion only: **6.69580s** (1.0345×).
- Exact FP32 weight/bias shadow only: **6.78295s** (1.0212×).
- Both: **6.55582s** (1.0566×).
- Native math and no vision shadow restored: **6.93489s** (+0.114% drift).

[Main summary](<summary.json>), [rank0 complete proof](<ranks/vision_rank0.json>)
and [driver](<vision_encode_tp8.py>) retain all8 rank receipts. This is a measured
**0.37118s** complete-stage saving, not proof that the remaining vision time is
all conversion overhead and not a claim of complete encoder≤2s. All modes remain
well above2s. The weight shadow adds2.20048GiB/rank for only an additional0.13998s
relative to activation-only in this experiment; production residency must evaluate
that tradeoff instead of assuming the largest cache is automatically best.
This is one fixed-order sweep with three warm observations per mode,not randomized
or interleaved sampling. The restored endpoint constrains end drift but does not
eliminate possible position/thermal effects;the independent profile control
reproduces the direction and similar relative magnitude.

All **160 complete encode_ids outputs** match the actual Round70 fresh reference,
including every call's embedding, fresh vision input/merged/three DeepStack
outputs, full text inputs/positions/mask and final CPU result. This comparison is
against the stable experimental FP32-compute stack, **not** independent HF,
original converted-FP16 production-model equivalence or official lossless video
quality.

## Scope and exact operation contract

The [instance-only wrapper](<vision_linear71.py>) binds116 actual vision Linear
modules:27 blocks×QKV/proj/fc1/fc2 plus the main and three DeepStack mergers×two.
Both variants preserve the original expression and every matrix dimension:

```python
F.linear(input.float(), weight.float(), bias.float()).to(input.dtype)
```

Activation-only replaces just`input.float()` with the verified native-layout
integer BF16→FP32 expansion. Shadow-only caches precisely the native
`weight.float()` and`bias.float()` results. No pretranspose/packing, GEMM merge,
external bias addition, BF16 output-cast change, LayerNorm arithmetic, patch
Conv3d, vision attention or vision RoPE modification occurs. Only methods on the
loaded vision instance are bound; the shared class and all text methods stay
unchanged.

All modes keep text shared-QKV/FP32-shadow projections,padded K/V,RoPE/SwiGLU,
Norm prefix/tail and activation expansion at the accepted fast configuration.
This is a **vision** factor A/B, not an accidental second text-math switch.

Actual input remains2,043 IDs, original FP16 transport pixels`[8064,1536]` and two
`[1,48,84]` grids. The normal encoder converts pixels to BF16. All752 actual
BF16 parameters from the original MiniMax-H3 checkpoint load, including real
vocabulary embedding and replicated vision. Inputs and complete parameter bytes
are hashed before/after and matched to Round70. The converted native-fp16
checkpoint and fake/zero embeddings are not substituted.

This diagnostic requires`VISION_FP32=1`; the driver sets it and the aggregator
pins all six precision/feature flags. The wrapper is not a general replacement
for the default BF16 vision path. Actual NPU runtime flags were measured, not
inferred from old reports: **matmul.allow_hf32=false,conv.allow_hf32=true**, identical
across all8 ranks before/after. We do not retrospectively claim these were recorded
in Round70.

## Actual numerical and native-layout proof

The three candidate first calls pass **2,784 candidate-vs-original vision Linear
output checks** (116×3 modes×8 ranks). Operand comparisons are separated by factor:
**1,856 enabled activation-expansion checks** plus928 native-conversion controls;
**3,712 enabled weight/bias-shadow checks** plus1,856 native-conversion controls.
The total recorded operand comparisons are2,784 input-cast and5,568 weight/bias,
but controls are not counted as independent enabled-factor proofs. Each covers
full raw bits,finite status,element counts and native shape/stride/physical format.
Actual GEMM dimensions remain:

- QKV weights`[3456,1152]`,output`[8064,3456]`.
- Attention projection weights`[1152,1152]`,output`[8064,1152]`.
- Block fc1 weights`[4304,1152]`,fc2`[1152,4304]`.
- Main/DeepStack merger fc1`[4608,4608]`,fc2`[5120,4608]`,2,016 rows.

The newly measured`[8064,4304]` activation has34,707,456 elements:8,473 complete
4096-element tiles plus one2,048-element tail. It is not inferred valid from the
earlier text shapes. Actual native stride/format equality is required;the wrapper
does not force contiguous
or convert format to evade unsupported-layout guards. Each original activation
has an explicit pre-cloned readonly check. Weight/bias immutability is instead
covered by the whole-model before/after byte hash, not falsely labelled a separate
per-call readonly proof.

First calls include reference double-compute and CPU proof copies and therefore
are **not warm component performance**. Their117–133s candidate totals
cannot be presented as isolated cache-construction cost;conversion,reference
work,compilation and component probes are mixed there. During the three warm
calls,all VERIFY and per-component PARTS synchronization paths are disabled.
The complete timer
retains real embedding/fresh vision/positions/DeepStack/text50 and the encoder's
native`hidden.cpu()` output plus ending drain. Start barrier/drain are outside;
extra intermediate copies and hashes are after timer stop. Retained diagnostic
references can change allocator lifetimes and are not a production peak budget.

## Exact shadow budget and restoration

The independent vision cache contains **232 weight/bias entries** and exactly
**2,362,750,400 bytes**. Each mode explicitly clears the previous cache before
constructing a fresh one. Shadow/both show232 misses+696 hits,zero invalidations
and zero uncacheable bypasses across first+three warm forwards. Native,
activation-only and restored contain0 vision shadow entries/bytes.

Allocated HBM snapshots/rank:21,525,576,192 bytes without vision shadow,
23,888,436,736 with it;restored returns to the former allocation. Reserved memory
remains27,902,607,360 bytes after restoration because the allocator retains blocks.
That is not a leak or a claim of peak usage/full DiT/VAE co-residency. The existing
text FP32 shadow still adds11.1084GiB/rank,plus1.9531GiB/rank padded K/V.

All instance methods restore and vision shadow bytes are0 at final exit. The
cache is built after full loading on versioned parameters. Descriptor guards do
**not** detect loader `.data` writes;clear remains mandatory before reload,
offload or unversioned updates. No production offload/load hook is installed.

[Main aggregator](<summarize_vision71.py>) independently pins the complete original
model/input,116 module names/geometries,the factor flags,all actual proofs,
strict cache policy,HF32 flags,whole outputs and restoration. Output differences
remain failed acceptance/counterexamples instead of being hidden by small RMS.
Complete HTTP preprocessing,input distribution,final FP16 broadcast,loading/
offload and full pipeline residency remain excluded and separately unproven.

## Independent internal vision event profile

[Profile driver](<vision_profile_tp8.py>) and [event wrapper](<vision_events71.py>)
use a new run root with native/both/restored clean-before/profile/clean-after
controls. Each mode performs one excluded first+three calls per phase, with no
same-input reference double-compute. Event pairs are preallocated/primed outside
measurement. No per-operator synchronization is added.

Each profiled vision has290 events:one inclusive parent,one patch embed,
one positional interpolation,one rotary-position construction,58 LayerNorm,
116 Linear,31 GELU,27 apply-RoPE and54 attention calls (two actual image segments
per each of27 blocks). The parent includes all children and must **never** be
added to their sum. Current-stream intervals include dependencies,host dispatch
gaps and stalls;they are not isolated occupancy/roofline measurements. Parent
minus flat children is unassigned and may be slightly negative from event
quantization;it is not silently clamped or labelled CPU/network.

[Profile aggregator](<summarize_profile71.py>) keeps clean wall,profile wall,
parent/flat-child and host-dispatch quantities separate,using each same-call
complete-wall slowest rank. It verifies all input/parameter/precision identities
against the main A/B rather than treating event collection as numerical success.

[Verified profile summary](<profile_summary.json>) and [same-call breakdown](<breakdown.json>)
contain **240 additional complete outputs**,all intermediate identities equal to
the main run,plus20,880 valid vision events. Across the two runs,400 actual fresh
complete encoder outputs match the stable reference. Clean-before medians in
this independent run are7.05376s native,6.70962s both,7.08923s restored. They are
reported separately,not cherry-picked or merged with the main five-mode results.
Cross-run absolute medians differ by about1.8–2.3%;only each run's paired relative
comparison is used to attribute the factor,not a cross-run subtraction.
Profile perturbations are+1.057%/+0.100%/+0.509%;clean-after drift is
+0.583%/+0.201%/-0.178%.

The representative both-mode profile is the median complete-wall call2/rank1:
complete encode_ids wall6.71634s,inclusive vision event **4.77220s**. The separate
parent host-dispatch wall is4.63456s;it is not interchangeable with that device
event interval. Its flat current-stream child sums are:

- Attention **1.98124s**.
- Linear0.75740s.
- Apply-RoPE **0.62273s**.
- GELU0.40088s.
- LayerNorm0.32730s.
- Patch embed0.13992s;position interpolation0.03641s;rotary positions0.00186s.

Flat children total4.26774s,and vision-parent-minus-children0.50447s is left
unassigned. These numbers are not summed with text time from a different run,
and host-dispatch values are not added to device durations. The profile now
supports prioritizing actual attention and vision RoPE rather than asserting
that further Linear casts/shadows can remove the remaining4.8s. It does not yet
localize attention's internal QK/softmax/PV or the unassigned remainder;those need
their own controlled probes before an operator-specific acceleration claim.

## Verification and final state

[Final fingerprints](<final_sources.json>) verify23 unchanged production/model/
dedicated-module files,20 executed diagnostic/dependency sources and the original
Round70 fresh-reference file. All match. [Cleanup receipt](<cleanup_receipt.json>)
records a final live ts-217 probe with8 cards idle,zero accelerator processes and
the owned lease released. No replacement server,production precision change or
CI pool use occurs.

[Regression receipt](<test_receipt.json>) records20 new tests plus124 prior tests,
**144 passing** with no skipped actual-data gates. It covers actual scope,math,
timer,cache,per-call layout/identity,profile parent/child and counterexample checks.
The actual numerical evidence remains the genuine NPU rank receipts,not CPU mock
simulations. Reproduce the portable gates with:

```bash
python3 -m unittest discover -s artifacts/diagnostics/round71 -p 'test_*.py' -v
python3 artifacts/diagnostics/round71/summarize_vision71.py
python3 artifacts/diagnostics/round71/summarize_profile71.py
```

## Unfinished acceptance

Stable experimental bit identity does not eliminate the independent multimodal
HF-reference gap. Earlier shared-visual CPU FP32-linear teachers and prior vision
RMS differences cannot be reclassified as full HF/lossless proof. No new49-step
quality generation,strict always-DiT<15s gate,full-A/V identity pass or official
T2VA/precision/warm-protocol matched result is claimed. Overall goal stays active.
