# Round79 diagnostic FP16 encoder candidate

**Status: root executed the FP16 vision subset on910A; see [actual vision evidence](<../vision.json>). The full text/vision encoder is not yet executed or precision-accepted.**

This directory holds a generated diagnostic variant of the production Qwen3-VL
encoder that removes the three BF16 I/O conversions which made the historical
half flag incoherent. It is **not** a production change and **not** a full-encoder
precision pass. The generation manifest records its pre-execution state;actual
runtime evidence is in the root probe receipt.

- Candidate: [candidate_encoder79.py](<candidate_encoder79.py>)
- Generator: [build_candidate79.py](<build_candidate79.py>)
- Manifest: [build_manifest79.json](<build_manifest79.json>)
- Static tests: [test_candidate79.py](<test_candidate79.py>) (25 passing)

## Provenance

The candidate is **generated, never hand-edited**. The generator reads the
Round76 production encoder read-only, refuses to run unless it hashes to
`7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d`, applies a
fixed list of 12 line-span edits, and writes the result. Regenerating from the
pinned source reproduces the candidate byte for byte, and any change outside a
declared span is impossible by construction:

```
source    7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d
candidate 2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78
edits 12   lines 1605 -> 1716
```

The manifest exposes the file and hash the root driver checks:
`candidate_path` and `candidate_sha256`. Production files are listed as
`production_files_modified: []` because none were touched.

## What changed and why

The Round78 audit ([npu_fp16_audit78.json](<../../round78/npu_fp16_audit78.json>)) found
that `VLLM_OMNI_910A_TE_DTYPE=half` only converted **parameters**: image and
video pixels were still cast to BF16 on the way in and the hidden state was cast
to BF16 on the way out, while several precision branches were BF16-gated and
therefore silently inert. That graph had never executed, and a single extra
hidden-state cast alone would have made any "FP16 alignment" measurement
meaningless.

| # | Edit | Reference-faithful behaviour |
|---|---|---|
| E0 | Module docstring | No longer claims "plain BF16 weights" |
| E1 | `_r79_compute_dtype` + embedding all-reduce | The single-device reference sums the whole vocabulary row in one lookup, so the FP16 collective is promoted to FP32 and rounded once, like the row-parallel projections already do |
| E2 | FP16 RMSNorm branch | FP32 reduction → cast the **normalised value** to FP16 → **FP16** gamma multiply. Folding gamma into the FP32 point math would round once instead of twice |
| E3 | FP16 text M-RoPE | FP32 frequency construction, same interleaved layout, `cos`/`sin` cast to the activation dtype exactly as the reference returns them |
| E4 | Position interpolation | FP32 bilinear weights and FP32 four-term sum for **both** diagnostic dtypes, converting once back to the parameter dtype |
| E5/E6 | Dtype resolution | One resolver is the single source of truth for parameters and I/O; recorded on the instance |
| E7/E8 | `pixel_values` / `pixel_values_videos` | Cast to the **stored vision parameter dtype**, not a hardcoded BF16 |
| E9 | Hidden-state return | Returns the graph's own dtype. The reference has no final BF16 hop; the hidden state is already the compute dtype, so the BF16 result is unchanged |
| E11 | Candidate constants | `R79_CANDIDATE` and the pinned origin SHA |
| E10 | `_r79_io_dtype` + `r79_self_check` | Fail-closed check that the graph really is one uniform parameter dtype |

### Deliberate deviation you must not mistake for a regression

**E4 makes BF16 interpolation FP32 too.** The production default only did this
under `VLLM_OMNI_910A_VISION_FP32=1`. Because the reference always interpolates
in FP32, the BF16 candidate is now numerically *more* faithful and is therefore
**not bit-identical to the production R76 BF16 default**. A BF16 A/B against
production must not expect bit equality in the vision position embeddings.

## What was deliberately *not* changed

- **Text SDPA call site** is byte-identical: the 910A shim's `±50000` clamp and
  FP32-softmax/FP16-GEMM decomposition remain a declared policy difference from
  the CPU reference's math backend. This is a measured-difference source, not a
  bug to "fix" inside the candidate.
- **Vision SDPA** keeps its existing FP32 QK/softmax/PV island, asserted equal to
  production. It happens to match the CPU gold's fp16 math-SDPA behaviour, but
  that agreement is measured, not assumed.
- **No global monkeypatching** and no R73 BF16 cast kernel reuse (that kernel is
  BF16-specific and would be silently wrong here).
- **BF16 branches preserved verbatim**, including `_NpuBF16CompatibleLinear` and
  `_NpuBF16CompatibleLayerNorm` which stay BF16-gated: FP16 takes the native
  operator rather than an unvalidated whole-FP32 shadow implementation.

### A delegation assumption that did not hold

`TEXT_PAD_KV`, `TEXT_FP32_LINEAR`, `_text_fp32_local_linear` and the
vision-FP32 shadow buffers **do not exist in this R76 source**; they belong to
the separate `round65/candidate_padded_text_encoder.py`. There was therefore
nothing to disable here, and the candidate does not introduce them. The static
tests assert their absence rather than silently pretending a change was made.

## Requirements for a future vision-only probe

These are properties of the source, asserted by the tests so they are not
rediscovered the hard way:

1. `MiniMaxH3Qwen3VLVisionModel(config)` takes only `config` — no text model and
   no encoder process group are needed.
2. `MiniMaxH3Qwen3VLVisionRotaryEmbedding._inv_freq` is a **plain attribute, not
   a registered buffer**. `.to()` never moves it, so a meta-device construction
   leaves it on meta. A probe must install a real CPU FP32 rotary module before
   the forward pass.
3. With FP16 parameters the FP32 interpolation path activates **automatically**;
   it does not depend on `VLLM_OMNI_910A_VISION_FP32`. Setting `TEXT_HF_MATH=1`
   and `VISION_FP32=0` does not disable it, and the module has no import-time
   assertion that would reject those flags.
4. Vision parameters keep their names and shapes, so per-tensor conversion
   hashes can be compared directly against the CPU reference's
   `parameter_conversion["visual.<name>"]`.

## What this does **not** establish

- No numerical pass threshold was selected by the probe;it reports actual CPU
  differences without declaring full-encoder acceptance.
- The Round78 CPU FP16 reference is a **separate** job. Its fresh vision tensors
  were consumed by the root vision probe;the complete reference's full repeat
  is still pending. Casting old BF16 outputs toFP16 is not a valid reference.
- Vision `nn.LayerNorm`/`Conv3d` and all27 blocks ran finite underFP16 on910A,
  with repeat-identical outputs. That proves this stage can execute,not that the
  complete50-layer encoder or the DiT's high-range activations have passed.
  The measured final vision relativeRMS difference is1.4568%;further same-input
  operator controls are needed rather than inventing a precision pass.
- Tensor-parallel row GEMMs still round partial outputs before theFP32
  all-reduce. That is a separate upcoming FP16 precision/kernel issue;the
  single-device replicated vision probe does not test it.

`r79_self_check()` is the intended fail-closed entry point: it raises if the
encoder is not one uniform parameter dtype, so a mixed BF16/FP16 graph cannot be
mistaken for a clean FP16 run.