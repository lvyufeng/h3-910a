# FastH3 (PDD-8) service bring-up on ts-87 — real Ascend910A

Target: MiniMax-H3 FastH3 release (`fasth3-inference-contract-v1`, PDD-8, ref2va)
running on 8x Ascend910A (32 GB each) as the **fast tier** behind OpenDirector.

Host: ts-87, CANN 9.2.0, driver 26.1.0.b083, torch 2.9.0, torch_npu 2.9.0,
vllm 0.18.0, vllm-ascend (python-only install with `soc_version=102 -> A2` patch),
vllm-omni editable install from `/data/h3fast/ref/vllm-omni` (0.30.0+npu).

## Launch

```bash
bash /data/h3fast/tools/serve_fasth3.sh     # logs: /data/h3fast/logs/serve_fasth3.log
```

Effective command:

```bash
vllm-omni serve --omni /data/h3fast/models/FastH3-Omni-FP16 \
  --task-type ref2va --trust-remote-code \
  --num-gpus 8 --tensor-parallel-size 8 --ulysses-degree 1 \
  --text-encoder-tp-size 8 \
  --diffusion-attention-backend TORCH_SDPA \
  --vae-use-tiling --vae-use-slicing \
  --host 127.0.0.1 --port 18091
```

Env: `VLLM_OMNI_910A_FLASH_ATTN=1`, `VLLM_OMNI_910A_TE_DTYPE=fp16`,
`VLLM_OMNI_910A_FP16_SEQ=1`, `VLLM_OMNI_910A_TEXT_HF_MATH=1`,
`VLLM_OMNI_910A_NO_FUSED_{ACT,NORM}=1`, FA package + kernel dir as in the script,
plus `LD_PRELOAD=<site-packages>/sentencepiece.libs/libgomp-*.so.1.0.0`.

## Why the shape of that command

- **Only 32 GB per card.** The released Perf-Preview DiT is 34.8 GB of FP8/FP16
  weights; dequantized FP16 needs TP8 (~4.4 GB/rank) before the 66 GB FP16
  encoder (TP8, ~8 GB/rank) and the ~10 GB video VAE share the same cards.
  TP2/USP4 OOMs while the DiT is still being constructed.
- **Attention backend name.** There is no `ASCEND_ATTN` enum on this tree; the
  910A chunked/FA kernel registers itself as the `TORCH_SDPA` implementation
  (`h3_910a_attn.H3910AAttentionBackend`) and is enabled by
  `VLLM_OMNI_910A_FLASH_ATTN=1`.
- **BF16 is not available in silicon on 910A.** `torch.bfloat16` matmul runs, but
  through an FP32 fallback; the encoder must stay FP16
  (`VLLM_OMNI_910A_TE_DTYPE=fp16`), which is also what the accepted A-machine
  path uses.
- **VAE tiling/slicing.** A 5 s 256x256 ref2va decode otherwise asks for a single
  16 GiB activation (~512 channels at 120x256x256 FP16) and OOMs at 32 GB.

## Environment fixes applied on the host

1. `vllm_ascend/ops/gdn.py`: the vendored qwen3.5 GDN patch imports `fla_npu`,
   which has no 910A build. The import is now optional and the GDN patches are
   inert for diffusion serving.
2. `vllm_omni/entrypoints/cli/__init__.py`: the trimmed source tree has no
   `vllm_omni.benchmarks` command; the benchmark subcommand imports are now
   optional so `serve` keeps working.
3. `vllm_omni/_version.py`: shim so the editable install does not warn about a
   missing generated version module.
4. `/data/h3fast/vllm_omni` (a stale copy of the tree) shadowed the editable
   install whenever the working directory was `/data/h3fast`; it was moved to
   `ref/vllm_omni_stale_20261128`.
5. `sentencepiece`'s bundled `libgomp` collides with the static TLS block once
   vLLM's server imports land; `LD_PRELOAD` that single library.

## Encoder Conv3d dtype mismatch

`MiniMaxH3Qwen3VLVisionPatchEmbed` builds an `nn.Conv3d` (1152 x 3 x 2 x 16 x 16).
On this install the loaded module ends with an FP32 weight and a half-precision
bias, so the reference code `self.proj(hidden_states.to(dtype=weight.dtype))`
sends FP32 input into a Half-bias ACL kernel and aborts every request with
`Input type (float) and bias type (c10::Half) should be the same`.

Fix (single small kernel, numerically the same math): run that conv in FP32 and
cast its output back to the module dtype (`encoder.py`,
`encoder.py.pre_910a_convfix` keeps the original).

## Status

- DiT load (585/585 tensors, zero missing) and PDD head preparation: OK.
- Service startup: OK, `/v1/videos` accepts ref2va requests with a JSON-string
  `image_reference` field (`{"image_url": "data:image/png;base64,..."}`).
- Encoder: OK after the Conv3d fix (FP16).
- Remaining: end-to-end short smoke (VAE tiling memory), then 480p/124 frames.