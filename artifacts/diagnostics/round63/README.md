# Round63 — VAE decode: actual loader parity, attention bottleneck and clean replay

**Main measured result:** self-developed TBE finite-select reduces complete
resident video-VAE decode from **20.648s to13.920s** without changing query chunking,
and to **11.590s** with chunk256. All128 full-video returns in that eight-card A/B
have the captured reference SHA256. These are warm resident decoder results for
one actual FL2VA latent, **not full service decode, original-BF16 quality or a
completed official lossless benchmark**.

This round continues the previous session's uncommitted decoder capture and
experiments. **No production optimization is installed.** Current comparisons
are against the actual native-FP16 910A runtime, not original BF16 inference or
an accepted official A2/A3 lossless-generation benchmark.

## Recovered evidence and a corrected baseline

The one-interval FL2VA [capture request](<e2e_capture/result.json>) used original
unfused weights, 1344×768, 5s/24fps/seed1234 and first/last references. Its aligned
output has124 frames. Capture hooks, CPU downloads and first-use compilation
make that request unsuitable as a clean latency benchmark. The production VAE
[capture restoration](<capture_restoration.json>) was already complete; the
continuation does not reinstall the hooks.

The old [tile replay](<recovered/tile_replay.json>) compared FP32 storage against
144 FP16 block Linears. Outputs agree with each other, but **neither matches the
captured production tile** (relativeRMS0.021605%, max absolute0.00390625). Its first
FP32 call29.18s versus warm~0.71–0.73s includes first-use work; it does not isolate
an autocast-cache speedup. The failed [old attention probe](<recovered/attention_probe.json>)
called unpatched native SDPA, had zero successful cases, and is retained as a failure.

The [second tile replay](<tile_replay02.json>) adds the actual worker platform
initialization and intermediate comparisons. It still diverges before the first
block, proving that merely setting internal formats does not repair the baseline.

The cause is a real loader contract: the remote
[from_pretrained](<source/minimax_h3_video_vae.py#L102-L111>) constructs the module
under the ambient dtype and copies weights into those parameters. The adapter
[later widens storage to FP32](<source/vae.py#L353-L358>). Thus **FP32 dtype metadata
does not prove original FP32 parameter values survived loading**. Service pipeline
construction uses default half; the standalone probe had explicitly used default
FP32. The encoder-only loader separately pins FP32 and must not be conflated with
this decode loader.

[Final teacher-forced replay](<tile_replay03.py>) constructs under half, restores
the ambient dtype in `finally`, and then uses the unchanged native adapter. Its
[receipt](<tile_replay03.json>) now matches actual layer0/17/35 norm inputs,
layer0 Q/K/V and the complete36-layer tile **raw-bit exactly**, including the
5,505,024 FP32 tile output elements. This is parity with current service state,
not a new claim that half-rounded VAE parameters equal the released FP32 values.

## The measured warm bottleneck is VAE attention

The [bounded first-two-tile trace](<profile_summary.json>) records rank0's second
tile at722.917ms:36 attention-core calls account for592.323ms, while145 norm calls
account for10.203ms and146 Linears for81.818ms. These are observed inclusive stream
intervals; capture/host gaps can contribute, so this is not a hardware roofline.
All ranks' full tile-call counts are retained, including the28/21 load imbalance.

VAE [Attention](<source/attention.py#L98-L123>) invokes the remote
[SDPA entry](<source/flash.py#L55-L75>), **not** the diffusion backend registry.
The actual [runtime shim](<source/runtime_sdpa_shim.py#L47-L95>) uses query chunks
of128, FP16 QK/scaled/clamped scores, FP32 softmax, then FP16 probabilities and PV.

The [corrected attention probe](<attention_probe02.py>) asserts that this shim is
installed, reconstructs actual Q/K/V strides (including the interleaved V view),
and includes its isfinite/where output handling. [Results](<attention_probe02.json>):
actual B1/S1797/H32/D64 attention is16.577ms at chunk128 versus15.285ms at chunk256.
All measured layout/chunk variants reproduce the actual attention output bits;
none justifies a large acceleration claim.

On the actual complete tile, warm medians from three synchronized calls are:

- chunk128:0.728141s;
- chunk256:0.675716s (**7.2% latency reduction**);
- chunk512:0.693238s.

All three [tile cases](<tile_replay03.json>) repeat exactly and match the real
captured output. Fixed ordering and three repeats are not a sustained-service
speedup or a multiplication into a whole-video timing.

## Exploratory FA is not silently accepted as lossless

The [padding/head partition probe](<attention_probe03.py>) retains the legacy
score/probability rounding in its eager candidates. All tested padding/partition
outputs match the captured attention, but none beats the simple chunk256 case.

The existing custom dense FA, padded2048 with exact valid-key limit1797, takes
10.135ms in that [separate-card run](<attention_probe03.json>) versus the same
run's16.098ms shim baseline. It differs from native in293,102/3,680,256 half words
(relativeRMS0.016063%, max absolute0.00390625). Although its error versus the
full-key FP32 oracle is slightly smaller than the legacy result for this fixture,
**closer on one operator is not full-video quality acceptance or bitwise identity**.
It is not installed in production.

### Raw tiling and packing: a useful rejected attribution

[Raw-NZ tiling](<attention_fa_tiling01.json>) improves the short-D64 kernel from
1.456ms (Br128/Bc256) to1.183ms (128/512) and1.061ms (64/1024), without packing
or output finite-select in those timers. Br32/Bc2048 aborts the compiler with an
L1 buffer-capacity error; the failure is retained and Br16/Bc2048 was not reached.
All successful FA variants still differ from legacy attention and remain rejected
as bitwise lossless replacements.

A self-developed [TBE pack/unpack builder](<pack_vector63.py>) and
[isolated wrapper](<packed_fa63.py>) fuse Q half scaling, zero padding, BSND to NZ
packing and inverse output conversion. On the [actual V-view fixture](<pack_validation01.json>)
all12,582,912 packed half words and3,680,256 unpacked values match Torch bits.
The same FA binary also produces identical finite outputs. However, the complete
wrapper is10.332ms versus10.316ms before packing fusion: **no material benefit**.
The raw/wrapper gap therefore must not be attributed to packing by subtraction.
The prepared-device object requires current device to remain fixed throughout its
calls; this diagnostic never switches away from card0.

## The real wrapper overhead: finite-select

[Stage isolation](<sanitize_validation02.json>) directly measures the original
`isfinite + where` output handling at8.550ms. The self-developed
[TBE finite-select kernel](<sanitize_vector63.py>) takes0.216ms, including the
noncontiguous BSND input's dense copy. Complete same-input legacy attention
falls from15.924ms to8.164ms, with every captured output half word unchanged.
This keeps QK/score/softmax/PV math and rounding unchanged; it is not a FA swap.

There is an important correctness boundary, retained rather than hidden:

- The kernel matches an integer-bit oracle for **all65,536 FP16 patterns**:
  preserve all63,488 finite words (including signed zeros/subnormals), select
  positive zero for2,048 NaN/Inf patterns.
- Under the **currently configured native NPU environment**, the old
  `isfinite + where` sequence returns the input unchanged for all65,536 words,
  including those2,048 nonfinite patterns. This is an observed configuration/path
  discrepancy, not a universal statement about every CANN/SoC mode.
- [First assertion-failing attempt](<sanitize_validation01.json>) is preserved.
  The corrected probe explicitly records the differing nonfinite behavior instead
  of changing its reference or claiming full native bitwise equivalence.
- The real captured raw attention output contains **zero nonfinite values**;
  old and new complete attention outputs match raw bits there. Full-video
  acceptance below is likewise scoped to the actual captured latent, not arbitrary
  pathological inputs or original BF16 generation quality.

The [isolated integration helper](<sanitize_integration63.py>) reads the actual
remote SDPA function and changes **only its checked final return expression**.
An AST comparison proves all earlier transposes, mask handling, backend context
and functional SDPA calls are unchanged. It restores the original function after
A/B. No model/checkpoint, global torch function or production file is patched.

## Clean complete decoder replay

[Eight-rank reproducer](<decode_replay_tp8.py>) uses the actual normalized video
latent, unchanged native spatial tiling, temporal blending, post-quant convolution,
pixel quantizer and HCCL tile gathers. Only candidate query chunk size and optional
FP16 materialization of144 block Linears change. A final FP32-storage/native-chunk
case restores the original service values. CPU output download/hash is outside
all timed intervals.

**Scope is resident video-VAE decode**: CPU model construction, initial H2D stage,
audio decoder, MP4 mux/delivery, text encoder and DiT are excluded. Cold first calls
are retained separately, not mixed into the warm median. The report takes a
per-call maximum over eight ranks, then a median over three warm calls. It never
uses maximum-of-rank-medians or divides E2E by step count.

The first complete [eight-rank A/B](<decode_replay/decode_replay01_rank0.json>) is
finished. The median of per-call rank maxima across three warm calls is:

- Baseline chunk128:20.5950s (first/cold call116.7225s, excluded).
- Chunk256 only:19.9841s; all three rank maxima19.9802–19.9926s, barely below20s.
- Chunk256 plus144 FP16 block Linears:20.0347s; no extra warm latency benefit.
- Original chunk/storage restored:20.5730s (−0.107% versus initial baseline).

All **128 complete outputs** (four modes×four calls×eight ranks), each383,975,424
uint8 bytes, have the same SHA256 as the real capture; no frame subsampling is
used. This is only about3% warm improvement from chunking, not a155s→20s controlled
speedup. CPU load and initial stage-to-device are retained separately per rank.

The newer [finite-select-only replay](<decode_sanitize_tp8.py>) additionally uses
exclusive run claims and atomic receipts, preserves FP32 parameter storage,
compares baseline/sanitizer/chunk256/restoration, and attempts restoration/group
cleanup before final publication. It also completes on all eight cards:

- Its own baseline:20.6476s; three per-call rank maxima20.6304–20.6543s.
- **Only TBE finite-select, chunk128 unchanged:13.9204s (1.483×)**;
  warm maxima13.7481/13.9204/14.2226s.
- **TBE finite-select plus chunk256:11.5897s (1.782×)**;
  warm maxima11.6041/11.5897/11.5857s.
- Original function/chunk restored:20.6482s (+0.00269% baseline drift).

Again, all128 complete video returns match the real captured reference SHA256;
all ranks restore the original remote SDPA function. Native and candidate shape,
uint8 dtype, full383,975,424-byte length and repeatability are checked independently
by the [aggregator](<summarize.py>), not accepted from `completed=true` or declared
`exact` flags alone. Both accelerated cases are below20s on every measured warm
rank-max call. The extra chunk256 gain is not attributed solely to the sanitizer.

[Verified summary](<summary.json>) and [rank0 raw evidence](<decode_sanitize/decode_sanitize01_rank0.json>)
retain cold/first calls separately. **No stage transfer/audio/MP4 time is added or
silently excluded from an alleged service result**: these measurements are only
the complete resident video decoder. Full service decode-stage≤20s, arbitrary
input/nonfinite behavior, original-BF16 quality and official-task acceptance remain
unproven.

## Reproduction, safety and remaining work

```bash
python3 -m unittest discover -s artifacts/diagnostics/round63 -p 'test_*.py' -v
python3 artifacts/diagnostics/round63/summarize.py
# Remote runs require a fresh owned lease and retained /tmp fixtures:
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
/home/mseco/vllm910a/bin/python -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29664 \
  /tmp/h3_round63_decode01/decode_replay_tp8.py
```

Use a new exact output/run ID; never replay a still-running or uncertain job.
Historical diagnostic scripts check for existing proof files but their snapshots
are not atomic locks. This session runs each only once and collects after process
completion. New [atomic evidence writer](<evidence_io63.py>) provides exclusive,
persistent run claims, atomic JSON revision replacement, and external-modification
detection for the new tiling probe. Its failure/ownership paths have local tests;
it does not retroactively turn older diagnostics into a concurrent service API.

Twenty-two portable reporting/AST/ownership/address-contract regressions and the
previous six whole-block regressions [pass](<test_receipt.json>); they are separate
from the retained
real-NPU arithmetic and complete-output checks. The
[final source verification](<final_source_verification.json>) checks18 unchanged
production/model/library sources and17 exact executed diagnostic sources.
All managed jobs have finished; eight cards were [live-probed](<cleanup_receipt.json>)
at0% with no accelerator processes and the owned lease was released. Model/latent/tile fixtures
remain at `/tmp/h3_round63_decode01` on ts-217 for a **freshly leased** follow-up.

Remaining: larger-lossless attention acceleration, actual full service decode
including transfers/audio within20s, stable text≤2s, whole49-step FL2VA quality
and stricter DiT margin, and a separately warmed task/precision/protocol-matched
T2VA comparison against official A2/A3. The overall goal remains active.

**Next acceptance gate, not more extrapolation:** integrate finite-select behind
an explicit default-off910A option with the nonfinite behavior documented, then
measure warm complete service decode (video+audio+residency transitions) using
unchanged native FL2VA settings. Keep this round's actual latent/capture as a
regression fixture. Follow with the native49-step request and a separate official
T2VA-matched warm request; do not relabel this one-interval latent's quality or
resident11.59s as either result. Production is currently still unmodified.
