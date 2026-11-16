# Round72 — exact partial vision RoPE tail; full gather prototype rejected

## Result and boundary

The complete resident encode_ids stage improves **6.58931→6.46305s (1.0195×)**
with the same-run native path restored at6.58284s (-0.0983% drift). The fixed
Round71 vision Linear and Round70 fast text stack stay unchanged. This is a
**0.12627s incremental saving**, not a complete text/encoder≤2s pass.

Three warmed per-call rank maxima:

- Native6.58931/6.54712/6.61798s.
- Partial tail6.45095/6.46305/6.46797s.
- Native restored6.56337/6.59025/6.58284s.

[Verified summary](<summary.json>), [rank0 full proof](<ranks/rope_rank0.json>) and
[full encoder reproducer](<vision_tail_tp8.py>) retain **96 complete outputs**
matching raw bits and all embedding/fresh vision/three DeepStack/text inputs/
positions/final CPU output identities. First candidate forwards pass all27
vision RoPE calls per rank: **216 pair checks /432 complete Q/K tensor checks**.
This is equality to the stable experimental compute path, **not** an independent
HF/full multimodal teacher,converted-FP16 production equivalence or official
49-step quality result. Outer HTTP processor,presentation,broadcast,loading/
offload and full pipeline residency remain outside this resident-stage timer.

This is one fixed-order native/candidate/restored sweep with3 warm calls each.
Restoration constrains endpoint drift but is not randomized/interleaved sampling.
Operator replay timings are reported separately and never substituted for the
complete stage or subtracted from timings in a different run.

## Why only partial fusion was enabled

Vision head width72 has a36-element half rotation. Full-row memory access is
aligned, but half36 is not a32-byte DMA unit. A genuine910A micro build of the
[full packed gather prototype](<vision_rope72.py>) fails before launch:

```text
current chip: Ascend910 AiCore doesn't support float32 for Instruction vgather
```

[Failure receipt](<gather32_failure.json>) and [current-platform support checks](<gather_support.json>)
are preserved. FP32 vgather was actually attempted and rejected;platform queries
also report false for tested float16/uint16/int16 alternatives. Those queries do
not prove that every possible shuffle is impossible,but no unsupported dtype
workaround or approximate rotation was enabled. The failed prototype remains
separate and is **never imported by the measured partial-tail path**.

The [enabled tail kernel](<vision_rope_tail72.py>) deliberately retains original
Torch `q.float()`,`k.float()` and `cat((-second_half,first_half))` rotation. Only
these operations move into oneTBE kernel per Q/K side:

```python
product0 = x * cos
product1 = rotated * sin
out = product0 + product1
out = out.to(torch.bfloat16)
```

The two FP32 multiply instructions and one FP32 add remain separate: no FMA,
reassociation,intermediate BF16 rounding,finite clamp or new signed-zero repair.
FP32 coefficients are loaded directly,not misread as BF16. Each72-float
coefficient row is288B=9 DMA blocks and is duplicated across16 heads in UB.
Each dense FP32 input row is1152 floats;the UB codec uses1280-element padding
but writes only1152 BF16 words. Half36 never appears in a DMA offset/length.
The existing integer BF16 RNE/pack is reused,not a BF16 ISA unavailable on910A.

The five data pointers are the exact four-input/one-output ABI:
`x,rotated,cos,sin,output`;current NPU stream is passed explicitly. Outputs are
independent dense`[N,16,72]` with stride`[1152,72,1]`,not overwritten inside
packed QKV. Actual `physical_format=2` is the ND format in this stack and matches
native outputs;it is not FRACTAL_NZ. Inputs are never forced contiguous to bypass
a unsupported-layout guard. The original Torch cast already produces the native
dense FP32 inputs required by this partial-tail kernel.

## Real fullpacked fixtures, not dense-copy stand-ins

The [rank0 capture](<capture/capture_rank0.json>) comes from one separately
excluded real complete encoder invocation with original already-FP16 transport
pixels,two48×84 image grids,the real original BF16 checkpoint and fresh vision.
All8 capture ranks complete and their full output equals the stable reference.
Selected vision blocks0/8/16/26 save **full packed QKV including V/neighbour
words**,not just a CPU-contiguous Q/K that erases pitch.

Actual manifests pin:

- Packed`[8064,3,16,72]`,55,738,368 bytes.
- Q/K view stride`[3456,72,1]`,offsets0/1152 BF16 words.
- Original FP32 cos/sin`[8064,72]`,stride`[72,1]`;two halves actually equal.
- Native BF16 Q/K outputs`[8064,16,72]`,dense stride`[1152,72,1]`.
- Current pointer alignment,dtype,storage bytes,physical format,complete tensor
  hashes,finite status and real maximum absolute values.

Four fixtures are97,544,537 bytes each,**390,178,148 bytes total**,under explicit
128MiB/file and512MiB total limits. Tensor blobs remain remotely retained;the
bounded [layer0](<capture/rope_layer00.json>),[layer8](<capture/rope_layer08.json>),
[layer16](<capture/rope_layer16.json>) and[layer26](<capture/rope_layer26.json>)
manifests and all capture receipts are committed. Capture copying/file IO is
excluded diagnostic work and is never included in a clean performance sample.

[Real replay validation](<validation01.json>) and [reproducer](<validate_real_tail72.py>)
load/reconstruct the full packed storage. Native replay first matches each saved
native output;candidate then matches all9,289,728 Q and all9,289,728 K values,
including native shape/stride/format,on every selected layer and3 warm repeats.
Reading the same storage with **wrong dense1152 pitch must disagree**;that
counterexample passes on every layer. Full packed source(including V) and reused
cos/sin hashes stay unchanged;an unaligned-but-contiguous coefficient is rejected.

Warm standalone wrapper medians,casts/original rotation included:

- Layer0:22.5355→16.8880ms.
- Layer8:22.4437→16.7757ms.
- Layer16:22.5474→16.9211ms.
- Layer26:22.4636→16.9101ms.

These are fixed-order standalone wrapper observations,not pure kernel timing or
a whole-stage substitution. The measured complete-stage saving is only0.12627s.

## Micro,loop and exceptional-value certification

[First partial-tail micro](<micro_tail01.json>) validates1/5/8-token dense output,
packed inputs,signed+0/-0 probes and native layout/bit identity. Its first cold
seconds include compilation/registration and are **not speed measurements**.

[Odd-row micro](<micro_tail02.json>) uses the actual runtime core budget **30**,
not an assumed32. N=2×30+1=**61** forces multiple steps and inactive last rows.
It also invokes the tail directly and proves all four FP32 operand buffers
`x,rotated,cos,sin` are readonly. Five compiled object/metadata receipts are
retained for1/5/8/61/8064 tokens,with validated blockDim and object hashes.

The finite RNE contract is not generalized to NaN payloads or FP32 overflow.
Actual captured tensor bounds are pinned far below the conservative2**120 margin,
and cos/sin≤1;all products/sums and actual outputs remain finite. Prior NaN/
overflow counterexamples remain reasons not to call this a generic Torch dtype
replacement. No new exceptional-value policy or CPU-teacher signed-zero repair
is introduced.

The original `encode_ids` entry **already has `@torch.inference_mode()`**. This
round does not turn an originally grad-enabled benchmark into a different
inference benchmark;the tail's inference-only guard is satisfied by the original
entry contract. A portable source regression pins that decorator.

## Timing,source and restoration controls

The complete timer preserves the prior boundary:start barrier/drain outside;
true embedding,fresh vision,positions/DeepStack,text50,original`hidden.cpu()` and
ending drain inside;additional proof/hash copies outside. First candidate calls
include all27 same-input native comparisons and compiler/cache setup and are
excluded from the three warm values. No capture hooks run in the timed A/B.

All8 ranks' matmul/conv HF32 flags remainfalse/true,and complete actual input and
752 BF16 parameter bytes remain unchanged. Vision Linear shadow stays
232entries/2,362,750,400 bytes in all RoPE modes,and text/padded-KV caches remain
unchanged. Allocated HBM snapshots stay23,888,436,736 bytes/rank and reserved
27,944,550,400;these are not peak usage or evidence of DiT/VAE co-residency.
The vision RoPE function,Linear wrappers and shadow buffers are restored/cleared
at exit. No production source is modified.

[Final verification](<final_verification.json>) checks23 production/model/
dedicated-module sources,24 executed diagnostic/dependency files,5 actual
reference/fixture files and5 compiled kernel object/metadata pairs. All match.
The [aggregator](<summarize72.py>) validates actual fullpacked bounds/offsets,
finite range,wrong-pitch and alignment controls,micro odd-row/readonly coverage,
all27 pair checks,full output/reference/source identities and the retained
gather rejection independently. Matching this reference does not mark the
whole goal complete. [Regression receipt](<test_receipt.json>) records15 new tests
plus144 prior tests,**159 passing** with no skipped actual-data gates. The new
checks cover exact instruction/ABI/row-padding contracts,original inference-mode
entry,fullpacked/V coverage,odd-row/signed-zero/native-layout proofs,retained
gather failure,fixture/source/object fingerprints and failed-identity controls.
[Cleanup receipt](<cleanup_receipt.json>) records all8 cards0% and accelerator
processes0 after all groups exited,and the owned lease released.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round72 -p 'test_*.py' -v
python3 artifacts/diagnostics/round72/summarize72.py
```

## Remaining work

Even erasing the prior0.623s RoPE category would not make6.56s complete encode2s.
Partial-tail fusion still pays packed cast and Torch rotate,and the last measured
vision attention category remains about1.98s. Next work should isolate attention
QK/softmax/PV and real cast/layout overhead before selecting a kernel or changing
precision. A full fused rotation needs a proven910A shuffle/packing primitive,
not the rejected vgather path.

Complete HTTP/residency timing,independent HF multimodal reference,strict
always-DiT<15s,full A/V identity and49-step task/precision/warm-protocol-matched
comparison remain unaccepted. The overall goal stays active.
