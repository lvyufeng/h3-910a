# MiniMax-H3 on Ascend 910A via vLLM-Omni

> **Current native/lossless optimization work:** see the
> [Round98 actual full TP8: NZ attention reduces corrected17.04%→9.92%/8.04%, fastest2.034s still not accepted](<artifacts/diagnostics/round98/README.md>),
> [Round97 direct-NZ softmax/packing: real leaves47ms→26.5ms, accurate vision1.825s; not native-bit lossless or full precision pass](<artifacts/diagnostics/round97/README.md>),
> [Round96 fused PV: exact all27/DeepStack/merger bits, real leaves~10% faster; batch-shape failure fixed and preserved](<artifacts/diagnostics/round96/README.md>),
> [Round95 actual8rank CPU-exact RoPE A/B is non-monotonic; corrected17%→13.6% only one case, fastest2.038s still not accepted](<artifacts/diagnostics/round95/README.md>),
> [Round93 CPU Half alone amplifies0.253%→17.16%, first15.8× jump at43; no NPU arithmetic needed to explain scale](<artifacts/diagnostics/round93/README.md>),
> [Round94 actual RoPE one-ULP differences; same-input rsqrt CPU exact and Newton worse, no redundant norm patch](<artifacts/diagnostics/round94/README.md>),
> [Round92 original CPU Half block25 SHA exact; same-input NPU error0.005–0.008%, no17% explosion at25; late propagation next](<artifacts/diagnostics/round92/README.md>),
> [Round91 actual full TP8: geometry cache bit-exact2.207s; vision fixes worsen full precision17%, fastest2.064s still not accepted](<artifacts/diagnostics/round91/README.md>),
> [Round90 real GELU None/tanh bug fixed with Float-vector erf; leaf error1000–4000× lower, joint vision1.367s/0.724% still not full precision pass](<artifacts/diagnostics/round90/README.md>),
> [Round89 actual patch+linear bias: vision error0.725%, resident1.381s; erf GELU gap still open, no full precision pass](<artifacts/diagnostics/round89/README.md>),
> [Round88 real vision attention error reduced6× at leaves; joint vision gap1.19% but3.047s, not default](<artifacts/diagnostics/round88/README.md>),
> [Round87 real vision bias error reduced31–51× at leaves; full vision gap1.29%, best native-row encode2.899s still not accepted](<artifacts/diagnostics/round87/README.md>),
> [Round86 CPU-layer24 rebase: late error0.52–0.72%; actual Half SiLU tiny error and Float variant identical](<artifacts/diagnostics/round86/README.md>),
> [Round85 actual PV error reduced15× at leaf; full attention/native-row CPU gap1.82%, combined path3.064s still not accepted](<artifacts/diagnostics/round85/README.md>),
> [Round84 actual FP32 row-HCCL TP8: CPU-condition gap2.19%, native gap7.58%, encode2.434s; not accepted](<artifacts/diagnostics/round84/README.md>),
> [Round83 portable FP32 bias error reduced26× at H3 KV size; four generic FP32 GEMM shapes executed](<artifacts/diagnostics/round83/README.md>),
> [Round82 CPU conditioning isolation: residual2.48%; actual FP16→FP32 TIK GEMM micro executed](<artifacts/diagnostics/round82/README.md>),
> [Round81 complete TP8 KV padding: repeat fixed, full encode2.214s, CPU FP16 precision gap7.42% remains](<artifacts/diagnostics/round81/README.md>),
> [Round80 CPU FP16 oracle verified; full TP8 repeat failed; small-N zero-padding stability candidate](<artifacts/diagnostics/round80/README.md>),
> [Round79 real FP16 vision probe: warm1.44s, CPU FP16 difference1.46%, full precision pending](<artifacts/diagnostics/round79/README.md>),
> [Round78 independent CPU FP16 reference: complete27+50 graph repeated bit-identically](<artifacts/diagnostics/round78/README.md>),
> [Round77 user-selected CPU FP16 acceptance: public FP16 ports found, independent gold to be built](<artifacts/diagnostics/round77/README.md>),
> [Round76 actual49-step videos: warm DiT14.66s/step, persistent logo artifacts, FP16 runtime](<artifacts/diagnostics/round76/README.md>),
> [Round74 complete independent original-BF16 HF reference: selected50 quality mismatch retained](<artifacts/diagnostics/round74/README.md>),
> [Round75 exact output-column/gather probes: narrow FC2 failure and separate padding recovery](<artifacts/diagnostics/round75/README.md>),
> [Round73 exact strided vision attention cast: complete encode6.464s→5.384s, not yet2s](<artifacts/diagnostics/round73/README.md>),
> [Round72 exact partial vision RoPE tail: complete encode6.589s→6.463s; unsupported gather retained](<artifacts/diagnostics/round72/README.md>),
> [Round71 exact vision Linear factors: complete encode6.927s→6.556s and measured attention/RoPE bottlenecks](<artifacts/diagnostics/round71/README.md>),
> [Round70 complete multimodal encoder A/B: 8.835s→7.050s, not yet2s](<artifacts/diagnostics/round70/README.md>),
> [Round69 exact resident text-only1.829s gate](<artifacts/diagnostics/round69/README.md>),
> [Round64 measured service decode and rejected whole-A/V acceptance](<artifacts/diagnostics/round64/README.md>),
> [Round63 VAE kernel investigation](<artifacts/diagnostics/round63/README.md>)
> and [native49-step baseline](<artifacts/diagnostics/round53/README.md>).
> The older four-step/low-step latency tables below use an additional FlashGen
> LoRA; they are **not** official-lossless native-H3 benchmark results.

Running [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) video generation on
Ascend **910A**-generation NPUs (`910B` without a suffix), 8×32 GiB, CANN 9.2.0,
through **vLLM-Omni** — and making it fast enough to serve.

**Result:** on a warm engine, a 10-second clip generates in **167 s at 640×384**
and **250 s at the native 1344×768** — inside a 5-minute budget — with host RAM
at 334 GB of 754 GB and HBM at 3.2 GB of 32 GB per chip.

| Clip | Resolution | Steps | Wall time (warm) | Budget 300 s |
|---|---|---|---|---|
| 10.12 s | **1344×768** (native) | 2 | **250 s** | ✓ |
| 10.12 s | **1344×768** (native) | 4 | 449 s | ✗ |
| 10.12 s | 640×384 | 4 | **167 s** | ✓ |
| 10.12 s | 640×384 | 4 | 219–287 s (older HSDP=1 config) | ✓ |

The checkpoint is a **4-step** distillation, so the 250 s figure runs 2 steps and
trades quality for latency: against the same seed the 2-step render is 12%
darker with 31% less frame-to-frame motion. The 4-step render is the
quality-preserving option at 449 s. Steady state is roughly 100–120 s per step
plus about 52 s of fixed encode/decode work.

The 768p path only exists because the DiT weights are shared through
`checkpoint_mmap` instead of copied per rank; see "Host memory" below.

**Cold starts are not small.** The first request after a restart pays one-time
kernel compilation plus the first full text-encoder and VAE weight transfer,
which measured 662 s for a 4-step 768p run against 449 s warm — a 1.5×
inflation, and encode/decode alone read 42 s/157 s cold versus 9 s/43 s warm.
Always warm up before measuring.

## Why this is not a straight port

The 910A generation is missing, or pathologically slow at, several things H3
assumes. Three findings drove the whole design:

**1. 910A has no native BF16 matmul.** A historical BF16-labelled software/
compatibility-path benchmark for GEMM
([11250,5376] × [5376,28672]) measured **1.01 TFLOPS**, versus
**170.57 TFLOPS in native FP16**—the former is not BF16 hardware throughput.
H3 ships BF16 weights; practical 910A generation uses FP16, while the recent
encoder experiment stores BF16 and computes through FP32/integer codecs.
Switching the old DiT path to FP16 took a layer from 14.7 s to 0.077 s, but
`DTYPE=half` changes the numerical graph and is **not proof of original-BF16
losslessness**.

**2. Fused attention does not exist.** SDPA, `npu_fusion_attention`,
`npu_prompt_flash_attention` all fail with `EZ1009 SoC version ascend910
verification failed`. Triton cannot help either: `bishengir-compile` has no
`Ascend910B` target. Attention is a chunked matmul/softmax implementation
(`artifacts/design/patches/h3_910a_sdpa_shim.py`).

**3. `nan_to_num` has no fp16 kernel.** `optype [NanToNum]` fails for fp16 while
fp32 and bf16 work. The video VAE calls it during decode. Note that patched
HuggingFace dynamic modules are regenerated from the model directory on every
load, so the fix has to live in the model source, not in
`~/.cache/huggingface/modules`.

## Host memory: how 768p fits at all

H3 is 62 GB of DiT plus a 63 GB Qwen3-VL text encoder. The naive deployment
copies each rank's weights into its own private heap, and with 8 ranks that is
**595 GB of a 754 GB box** — measured 74 GB per worker with PSS equal to RSS, so
genuinely unshared. 768p then pushed host RAM to 715/754 GB, the box swapped,
and an ALLGATHER stalled into a deadlock.

The fix is `checkpoint_mmap`: keep the weights as read-only views of the
checkpoint so the 8 ranks share one page cache, instead of flattening a private
copy each. Two conditions blocked it, and clearing them is the whole recipe:

1. **HSDP must be off.** `build_checkpoint_mmap_plan` rejects `use_hsdp`.
   Dropping it costs nothing here — 640×384 actually got *faster* (219–287 s →
   167 s), because the mmap savings beat the weight sharding.
2. **The checkpoint must already be fp16.** mmap is zero-copy, so file dtype
   must equal runtime dtype, and H3 ships BF16 while bf16 GEMM is 168× slower.
   Convert offline, once; converting on load defeats the purpose because the
   converted copy becomes anonymous private memory again.

```bash
# Convert ONLY the DiT to fp16 (F32 tensors stay F32), then link the rest.
python artifacts/design/patches/convert_dit_fp16.py \
    /path/to/MiniMax-H3-4step /path/to/MiniMax-H3-fp16
```

| | HSDP=1 (old) | HSDP=0 + fp16 mmap |
|---|---|---|
| Host RAM per worker | 74 GB | **2.9 GB** |
| Host RAM total | 595 GB | **29 GB** |
| HBM per chip | 30 / 32 GB | **3.2 / 32 GB** |
| 768p 10 s | deadlock | **543 s** |

## Quick start

Requires a 910A host with CANN installed, vLLM 0.30.0 built with
`VLLM_TARGET_DEVICE=empty`, plus `vllm-ascend` and `vllm-omni` 0.30.0.

```bash
# 1. Fix the model's VAE: replace its fp16 nan_to_num call with an
#    isfinite/where equivalent. This must edit the MODEL DIRECTORY, not the
#    HuggingFace module cache -- see docs for why.
python artifacts/design/patches/patch_model_vae_nan_to_num.py \
    /path/to/MiniMax-H3-4step

# 2. Convert the DiT to fp16 so the mmap weight path can be used (see above).
python artifacts/design/patches/convert_dit_fp16.py \
    /path/to/MiniMax-H3-4step /path/to/MiniMax-H3-fp16

# 3. Serve. HSDP=0 is what unlocks the shared-memory path.
HSDP=0 TE_TP=8 MODEL=/path/to/MiniMax-H3-fp16 DTYPE=half \
  bash runtime/omni-serve-910a.sh --daemon

# 4. Warm the engine first -- the first request pays a large one-time cost
#    (kernel/shape warmup); without this step measurements are meaningless.
curl -s -X POST http://127.0.0.1:9198/v1/videos \
  -F "prompt=warmup" -F "seconds=4" -F "size=640x384" \
  -F "num_inference_steps=4" -F seed=0

# 5. Generate. num_inference_steps=2 is what fits 768p in the 5-minute budget;
#    use 4 when you want the checkpoint's full quality (449 s).
curl -s -X POST http://127.0.0.1:9198/v1/videos \
  -F "prompt=a red fox running through snow, cinematic" \
  -F "seconds=10" -F "size=1344x768" -F "aspect_ratio=16:9" \
  -F "num_inference_steps=2" -F seed=1
# 250 s warm -> {"id": "video_gen_..."}   then poll GET /v1/videos/{id}
```

## Cold start vs warm engine

The gap is large enough that it dominates any benchmark:

| Scenario | 10 s clip |
|---|---|
| First request after server start | 894 s |
| Warm engine | **219–287 s** |

The first denoise step costs ~45 s (`phase pre=16.1s blocks=29.4s`) while steady
state is 7.5 s, and the first VAE decode compiles kernels for that exact
resolution. Always warm up before measuring.

## Where the time goes (10 s clip, warm)

```
denoise (4 steps)     96 s    prep 1.8 s + model 7.5 s + sampler update 3.0 s per step
vae decode + mux     ~120 s
text encode           ~14 s
```

Inside the model, the 50 DiT blocks are 6.9 s and preprocessing 0.6 s. The
remaining per-step cost is sampler bookkeeping and sequence-parallel traffic.

## Layout

```
runtime/     launch script (start here: omni-serve-910a.sh) + HCCL probes
scripts/     model merge/LoRA conversion, NPU probes, protocol + smoke helpers
docs/        engineering log, and the 910A operator/op notes
artifacts/   the patches that make the stack run on 910A, plus its probes/tests
```

`docs/vllm_omni_910a.md` is the detailed multi-round engineering log, including
the approaches that did **not** work and why.

## Known gaps

- **int8 quantization is not wired up.** vLLM-Omni ships only an fp8/HSDP
  quantizer and 910A rejects the dynamic-quant parameters. `QUANT` defaults to off.
- **Text encoder is streamed** from host memory on every request (~14 s). Keeping
  it resident instead exhausts host RAM, because eight ranks each stage a full
  copy before sharding.
- **VAE decode dominates** the remaining time and still runs partly rank-locally:
  the decoder splits 640×384 into 6 tiles, which does not divide 8 ranks, so it
  falls back to every rank decoding everything. Shrinking the tile to force 8
  tiles was tried and is **1.5× slower** — the overlap recompute outweighs the
  parallelism.
- **First request is slow.** A warmup at server start would hide it.

## License / upstream

Patches here apply to vLLM-Omni and MiniMax-H3, each under its own license. The
upstream H3 checkout is not vendored; see `github.com/MiniMax-AI/MiniMax-H3`.