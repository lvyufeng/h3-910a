# Round 50 — BF16-preserving vision compatibility on 910A

## A working norm path, without an FP16 model conversion

The prior real FL2VA request stopped at BF16 vision LayerNorm: the installed TBE
LayerNorm implementation cannot compile its internal BF16→FP32 cast for this SoC.
The separate torch BF16↔FP32 casts **do** work and preserve the tested values.

The first composition (`FP32 cast → native FP32 LayerNorm → BF16`) ran, but failed
its numerical check on a constant input row: eps=1e-6 amplifies small mean errors
into ~1e-3 output error. It was **not deployed**. The
[first probe receipt](<layernorm_probe01_result.json>) preserves that failure.

The selected compatibility class centers before reducing and then computes a
second-pass variance in FP32:

```text
x32 = x.float()
shifted = x32 - x32[..., :1]
centered = shifted - mean(shifted)
variance = mean(centered²)
y = centered * rsqrt(variance + eps) * weight.float() + bias.float()
return y.to(original_dtype)
```

This retains BF16 parameter storage and BF16 I/O, including tested values above
FP16's finite range. It is the same normalization formula with a stable arithmetic
order, **not a guarantee of bitwise identity to every backend**, or a proof over
all possible extreme BF16 inputs.

Against an explicit float64 centered-variance reference at the model's actual
vision widths 1152 and 4608:

- six normal/constant/zero/high-offset cases all remain finite;
- constant and zero rows are **exactly bias**;
- maximum relative RMS error: **2.896e-5**;
- maximum absolute error after BF16 rounding: **0.0078125**;
- small-probe latency: approximately 1.05–1.12 ms, **not encoder-stage time**.

[Stable probe](<layernorm_probe_stable.py>),
[raw results](<layernorm_probe02_result.json>), and
[numerical summary](<numerical_summary.json>) are reproducible.

## The complete vision block exposed a second, independent defect

The patched norm alone was not enough. Native unmasked vision SDPA (head_dim=72)
failed with `can not cast format when output is input`. Three controlled attempts
preserve the same failure:

1. original BF16 SDPA;
2. explicitly contiguous BF16 inputs;
3. explicitly contiguous FP32 inputs with BF16 result casting.

This is an implementation/format-alias defect, **not evidence that contiguous
copies or an FP16 precision switch solve it**, and not a hardware impossibility.
Receipts: [attempt 1](<vision_block_probe01.json>),
[attempt 2](<vision_block_probe02.json>), [attempt 3](<vision_block_probe03.json>).

The selected NPU vision route uses the explicit unmasked attention formula
`softmax(Q @ Kᵀ / sqrt(D)) @ V`, with FP32 tensors for BF16/FP16 inputs and original
output dtype. There is no mask, dropout, approximation, or sparse-attention change
in this vision call. CPU/CUDA still delegate to native SDPA. This route does not
alter the separate DiT 30-core FlashAttention implementation.

The exact patched complete vision block now runs on NPU with BF16 weights and
BF16 output. On a synthetic `(128,1152)` input, relative RMS difference from the
CPU BF16 block is **0.002013 (0.2013%)**, max absolute difference 0.03125, all finite.
This is a component numerical check, **not** a full H3 quality or official lossless
assessment. The BF16 rounding and backend arithmetic are not bit-exact.

[Successful block receipt](<vision_block_probe04.json>),
[smoke reproducer](<vision_block_probe.py>), and
[attention regressions](<test_vision_attention.py>) are provided.

## Regression/deployment evidence

**21 final CPU/NPU regressions pass**: 9 normalization, 9 existing masking, and
3 vision attention tests. Coverage includes constant/zero and high-offset rows,
wide/postshuffle normalization, multi-axis/noncontiguous inputs, identical
checkpoint parameter keys/loading, CPU native branches, irregular Q/K lengths,
head_dim=72/128, and a uniform-attention case. The first harness version itself
called native BF16 `IsFinite` on NPU, which this driver rejects; finite checks were
moved to copied CPU results and a fresh process reran the tests. That harness
failure was not ignored or attributed to the norm implementation.

[Exact candidate](<candidate_encoder.py>), [norm tests](<test_layernorm.py>),
[mask regressions](<test_embedding_mask.py>), and the
[portable patch](<../../design/patches/encoder_bf16_layernorm.patch>) are included.
The portable patch is applied to a fresh baseline copy and verified byte-for-byte
against the tested candidate before publication. Parameter names and BF16 storage
are unchanged; the old encoder is preserved in a checked exclusive backup.

The deployed SHA256 is
`9d25561d952b910a9202a4241ece4075a149b0c8eab8c20f1e4b4e532614abe1`.
The original [deployment receipt](<deployment.json>) has an outdated scope label
("vision LayerNorm only") because the script label preceded the subsequent SDPA
repair. Its before/after hashes cover the **actual final norm-and-attention
candidate**; the reproducible deployment script now describes the full scope.
No model weights, inference schedule, TE_DTYPE flag, or power/clock setting changed.

## Full real-workload validation

A fresh 8-rank engine after the tested repair ran the same verified request form
as round 49: explicit `seconds=5`, 1344×768, fps=24, seed=1234, and actual first/last
images. Startup took 245.04 s. It passed the former vision runtime errors and
reached conditional-image VAE encoding, then failed with:

```text
Current operator reflection_pad3d do not support internal format.
```

The [failed request receipt](<e2e01/result.json>) reports 534.68 s of request
processing before failure. This is **not** a successful encode time or video
latency. No denoise/DiT timing or video output exists. In the inspected pipeline,
`encode_prompt` returns before `_encode_local_media`; the traceback stops in
`video_vae.encode_image → Downsample3D.forward → F.pad`. Passing a runtime point
still does not prove full-model numerical quality.

The padding call is at [vae_cnn.py:79](<source/cached_vae_cnn.py#L65-L80>) and uses
`(0,1,0,1,0,0)` in reflect mode on a tensor with an internal NPU storage format.
The cached and canonical VAE source bytes are identical; paths and hashes are in
[the next-repair receipt](<vae_padding_location.json>). The next round should
validate an exact conversion to a base ND/NCDHW format before this pad, rather
than globally disabling internal formats and penalizing every convolution.

The current repair is functional compatibility, not a completed performance
optimization: no 15 s DiT, 2 s encoder, 20 s VAE or official lossless comparison is
claimed. The norm and vision-attention fixes remain deployed with their original
encoder backed up. All managed engine processes are stopped, all eight cards are
verified without processes, and the fleet lease is released at round completion.
