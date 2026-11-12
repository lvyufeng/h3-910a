# MiniMax-H3 on Ascend 910A via vLLM-Omni

Running [MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3) video generation on
Ascend **910A**-generation NPUs (`910B` without a suffix), 8×32 GiB, CANN 9.2.0,
through **vLLM-Omni** — and making it fast enough to serve.

**Result:** a 10-second 640×384 clip generates in **219–287 s** on a warm engine,
against a 300 s budget.

| Clip | Resolution | Frames | Wall time |
|---|---|---|---|
| 10.12 s | 640×384 | 243 @ 24 fps | 219 s |
| 10.12 s | 640×384 | 243 @ 24 fps | 287 s |

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

## Quick start

Requires a 910A host with CANN installed, vLLM 0.30.0 built with
`VLLM_TARGET_DEVICE=empty`, plus `vllm-ascend` and `vllm-omni` 0.30.0.

```bash
# 1. Fix the model's VAE: replace its fp16 nan_to_num call with an
#    isfinite/where equivalent. This must edit the MODEL DIRECTORY, not the
#    HuggingFace module cache -- see docs for why.
python artifacts/design/patches/patch_model_vae_nan_to_num.py \
    /path/to/MiniMax-H3-4step

# 2. Serve.
bash runtime/omni-serve-910a.sh --daemon

# 3. Warm the engine first -- the first request pays a large one-time cost
#    (kernel/shape warmup); without this step measurements are meaningless.
curl -s -X POST http://127.0.0.1:9198/v1/videos \
  -F "prompt=warmup" -F "seconds=4" -F "size=640x384" \
  -F "num_inference_steps=4" -F seed=0

# 4. Generate.
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
runtime/     launch and control scripts (start here: omni-serve-910a.sh)
scripts/     model merge, benchmarking, probes, server helpers
docs/        engineering log, operator notes, API reference, acceptance criteria
config/      extra model path config
artifacts/design/   patches and probes that make the stack run on 910A
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