# Round70 — real complete multimodal encode_ids, not the resident text microstack

## Result and remaining target

The complete resident **encode_ids** stage improves from **8.83489s to7.05001s
(1.253×)**. It includes true vocabulary embedding, freshly recomputed two-image
vision, placeholder/position/DeepStack preparation, all50 text layers and the
method's final **CPU output transfer**. It is still far above2s. The prior1.829s
result is text-only with already prepared visual inputs; it is **not** this stage.

Three per-call rank maxima:

- Same-run experimental baseline:8.83436/8.83489/8.84945s.
- Fast exact text kernels:7.01965/7.08655/7.05001s.
- Original text math restored:8.86830/8.81912/8.87072s.

All **96 complete encode_ids outputs** match raw bits across mode/rank/repeat.
Every call also retains embedding IDs/output, fresh vision inputs and complete
merged+three DeepStack outputs, text inputs/positions/mask/DeepStack, layer50
output and the CPU result. All seven identity categories match across the whole
run, so no cached vision output or old teacher-forced features replace the real
vision forward. Fast's first call additionally passes **1,600 Norm output,
400 residual,1,600 activation-cast and2,400 projection checks**.

[Complete verified summary](<summary.json>), [main rank0 proof](<ranks/encoder_rank0.json>)
and [complete reproducer](<complete_encode_tp8.py>) retain the evidence. Matching
this same-run experimental reference is **not** independent BF16 HF correctness,
original production FP16-source equivalence or official49-step video quality.
The complete-service text≤2s and overall lossless objectives remain unaccepted.

## Exact inputs, weights and scope

- The input fixture is the original captured `encoder_input.pt`:2,043 token IDs,
  FP16 `pixel_values[8064,1536]` and image grids `[[1,48,84],[1,48,84]]`. Optional
  video inputs are actuallyNone and are not dereferenced or substituted.
- The pixel fixture **already includes production FP16 transport rounding**;
  `_encode` then casts to BF16 as before. We do not feed a newly generated FP32
  processor tensor and call it the same input. The original fixture file SHA and
  every actual tensor identity are pinned.
- Checkpoint source is the original **MiniMax-H3 BF16** text/vision directory,
  not the converted native-fp16-audit53 source. All752 real parameter tensors
  load, including the real vocabulary embedding and replicated full vision.
  There is no zero embedding or `nn.Identity` substitution.
- BF16 complete encoder parameters occupy7,480,909,280 bytes/rank:vision
  1,190,533,600;vocabulary embedding194,478,080;all text including embedding
  6,290,375,680. All model parameter bytes are hashed before and after.
- Both variants keep `VISION_FP32=1`, experimental HF-math Norm/FP32 projections,
  native-round TBE RoPE/SwiGLU, FP32 causal attention,1024-output padded K/V,
  exact FP32 shadow weights and the same shared-activation geometry. Only the
  fast variant enables the proven text Norm prefix/tail and BF16 activation
  expansion. Vision formulas/classes are not monkeypatched by text hooks.

The public `encode_ids` call **includes `hidden.cpu()`**, so that transfer stays
inside this timer. Outer HTTP image/video processor, prompt/presentation,
encoder-group input distribution, final FP16 hidden broadcast/H2D and component
load/offload are **excluded**. The full production service also uses its actual
residency schedule and checkpoint precision; those need a separate deployment
A/B. This is neither `encode_prompt` HTTP time nor the old unpartitioned
“encode+conditioning” aggregate.

## Independent warm component probes: fresh vision now dominates

A second [probe reproducer](<warm_components70.py>) uses a **new run root** and the
same full model/input. It performs first(excluded),3 clean-before,3 synchronized
component probes and3 clean-after forwards per mode, with no same-input reference
double-computations during probes. These **240 additional complete outputs** and
all intermediate identities match the main run, making336 complete outputs across
the two experiments.

[Probe summary](<component_summary.json>), [probe rank0](<components/encoder_rank0.json>)
and [same-call breakdown](<breakdown.json>) report clean medians:

- Baseline8.62562s;fast6.89760s;restored8.63728s.
- Fast clean-after6.92075s;probe wall6.93873s median.
- Probe perturbation relative to clean-before:+0.106% baseline,+0.596% fast,
  -0.0507% restored;clean-after drift:-0.181%/+0.336%/-0.330%.

The representative fast probe is the actual median complete-wall call1/rank5:

- Embedding0.02557s.
- Fresh vision **5.04089s**.
- Complete text model1.82829s.
- Probe wall6.93873s;sum of those component walls6.89476s;0.04398s stays
  unassigned rather than being inventively labelled pure CPU or network.

These are **synchronized component-probe intervals**, not clean component timings
or a mixed-rank/additive HTTP attribution. Each representative uses the slowest
complete-wall rank from that same call; per-component maxima from unrelated
ranks are never summed. First component values in the original run include cold
compilation/cache state and fast reference double-compute and are explicitly
excluded from warm attribution. The new probe, not those first values, supplies
the actual warm component evidence.

## Timing, identity and memory controls

Start barrier and device drain precede each complete timer; the model plus ending
drain and native CPU result are inside. Extra proof copies and hashes occur only
**after** the complete timer. Wrappers retain live per-call tensors, not timed CPU
copies, and release references in`finally`. Those retained diagnostic references
can affect HBM lifetime and allocator behavior; this is not a production peak
memory claim. Both before/probe/after paths retain the same references.

Main allocated HBM snapshots are21,525,576,192 bytes/rank;reserved snapshots
25,341,984,768→25,344,081,920 bytes/rank. Existing FP32 weight shadows still cost
**11.1084GiB/rank**, plus1.9531GiB/rank padded K/V. They fit this isolated full
encoder but do not prove co-residency with DiT/VAE or successful offload lifecycle.
The restored mode restores mathematics only;the same shadow buffers stay resident.

The [main aggregator](<summarize.py>) independently validates real input shapes/
dtypes/SHA,actual BF16 checkpoint index and parameter footprint, fixed precision
flags,all first-call same-input checks, per-call multimodal identities and final
output. It preserves a false whole-identity result if outputs differ, rather
than turning a successful HTTP/runtime status into numerical acceptance. The
[probe aggregator](<summarize_components.py>) requires parameter/input/checkpoint/
flag consistency with the accepted main run and reports clean/probe effects
separately. It does not convert matching output hashes into independent HF quality.

## Fresh reference fixture and next gate

The warm run saves a207,322,943-byte **fresh actual** encoder reference fixture:
real embedding output, newly recomputed vision merged/DeepStack, complete text
input and baseline complete output. [Final verification](<final_verification.json>)
checks its file/tensor hashes against the actual first baseline, and pins23
unchanged production/model/dedicated-module plus17 executed diagnostic/dependency
sources. This fixture is an experimental replay reference, **not an HF teacher**.

A fresh, independent HF vision→embedding/positions/DeepStack→50-layer language
reference using the same original processor/transport tensors is still needed.
Prior CPU FP32-linear teachers with old shared native visual inputs, and the
7.22% experimental language discrepancy or5.66% earlier vision RMS, cannot be
relabelled as an unchanged BF16 HF/full multimodal pass.

The next speed experiment should target the measured5s fresh vision path while
preserving its exact same-input tensor contract: first profile/cast/weight-cache
changes at unchanged GEMM geometry, then actual full vision and complete encoder
identity. Do not continue polishing resident text and call7s complete encode2s.
Complete service residency/processor/broadcast, full A/V identity, strict DiT<15s,
49-step quality and official task/precision/warm-protocol matched comparison all
remain separate open gates. The overall goal remains active.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round70 -p 'test_*.py' -v
python3 artifacts/diagnostics/round70/summarize.py
python3 artifacts/diagnostics/round70/summarize_components.py
```

[Regression receipt](<test_receipt.json>) records16 new tests plus108 prior tests,
**124 passing** in total. New tests exercise the actual complete timer, real
loader/no-fake-embedding scope, original input precision, same-input proofs,
nonhex/wrong-shape/missing-vision rejection, preserved output counterexamples,
precision/parameter-footprint guards and same-call component aggregation. These
portable AST/mocks/receipt tests are not synthetic substitutes for actual NPU
numerical execution. [Cleanup receipt](<cleanup_receipt.json>) records all8 cards
idle/processes0 and the owned ts-217 lease released. No production source or
replacement service was modified or left running.
