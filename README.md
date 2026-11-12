# MiniMax-H3 on Ascend 910A via vLLM-Omni

Running [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) video generation on
Ascend **910A**-generation NPUs (`910B` without a suffix), 8×32 GiB, CANN 9.2.0,
through **vLLM-Omni** — and making it fast enough to serve.

**Result:** on a warm engine, a 10-second clip generates in **167 s at 640×384**
and **543 s at the native 1344×768**, with host RAM at 331 GB of 754 GB and HBM
at 3.2 GB of 32 GB per chip.

| Clip | Resolution | Frames | Wall time (warm) |
|---|---|---|---|
| 10.12 s | 640×384 | 243 @ 24 fps | **167 s** |
| 10.12 s | **1344×768** (native) | 243 @ 24 fps | **543 s** |
| 10.12 s | 640×384 | 243 @ 24 fps | 219–287 s (older HSDP=1 config) |

The 768p path only exists because the DiT weights are shared through
`checkpoint_mmap` instead of copied per rank; see "Host memory" below.
Cold starts add roughly 1000 s of one-time kernel compilation — always warm up
before measuring, or you will read a 2.9× inflated number.

## Why this is not a straight port

The 910A generation is missing, or pathologically slow at, several things H3
assumes. Three findings drove the whole design:

**1. bf16 matmul is crippled — 168× slower than fp16.** The same GEMM
([11250,5376] × [5376,28672]) measures **1.01 TFLOPS in bf16** and
**170.57 TFLOPS in fp16**. H3 ships in bf16, so the naive port spends ~99% of its
time waiting. Switching to fp16 took a single DiT layer from 14.7 s to 0.077 s.
`DTYPE=half` is mandatory, not an optimization.

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

# 5. Generate.
curl -s -X POST http://127.0.0.1:9198/v1/videos \
  -F "prompt=a red fox running through snow, cinematic" \
  -F "seconds=10" -F "size=640x384" -F "aspect_ratio=16:9" \
  -F "num_inference_steps=4" -F seed=1
# -> {"id": "video_gen_..."}   then poll GET /v1/videos/{id}
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