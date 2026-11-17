# Round80 TP8 FP16 encoder diagnostic driver — root execution started

The fresh-fork author delivered this as **NOT EXECUTED** source. The root
reviewed it,corrected the mistaken batch-shape defect claim below,and launched
it under a bounded supervisor. Execution evidence is separate from the source
contract;no precision or speed pass is claimed before collecting all8 receipts.

| Item | Value |
|---|---|
| Driver | [`encode_fp16_tp8_80.py`](<encode_fp16_tp8_80.py>) |
| Static tests | [`test_driver_static80.py`](<test_driver_static80.py>) — stdlib only, no torch import |
| R79 candidate imported | `candidate_encoder79.py`, SHA256 `2141b29adc9ecc601d7fa2599e6ba56351adfbdaa5f16acb62c00d95cee3cf78` |
| CPU FP16 gold consumed | Round78 `reference.json` + `FP16_vision.pt` / `FP16_multimodal_inputs.pt` / `FP16_selected50.pt` |
| CPU reference source | `50165e77adb393b4573fbe0fdb29c62a6ddb14b8f39082d0ea67e904848010ea` |
| Original BF16 parameter SHA | `a69676bac5e6f3a513ce44080d29c73974553b6e0a327ea272e47b600d84e46e` |
| Fixture | `/tmp/h3_round54_capture01/encoder_input.pt`, `ed099e79…`, 2043 tokens, 8064×1536 FP16 pixels, grids `[[1,48,84],[1,48,84]]` |
| Planned private root | `/tmp/h3_round80_full_fp16_01` (created by the root; this driver only writes inside it) |
| Ranks | exactly 8 (`torchrun --nproc_per_node=8`) |
| Warm timed calls | 3 per rank, each including the final device→host copy |

## What the driver measures

Per rank, against the independent CPU FP16 recomputation (never an old BF16
output that was cast down):

- fresh vision merged features and all 3 DeepStack features;
- vocabulary embedding hash;
- text-model inputs: `inputs_embeds`, `position_ids`, `visual_pos_masks`,
  `deepstack_visual_embeds`;
- all **50** raw pre-DeepStack decoder-layer boundary hashes, plus full tensor
  comparison for the six layers the CPU reference saved (`0,1,2,9,24,49`);
- the selected-50 **pre-final-norm** hidden state.

Structural assertions: uniform FP16 parameters, finite output, bit-identical
repeat forward, and unchanged parameter/input hashes before and after.

Non-assertions, deliberately: there is **no** numerical threshold, no
`numerical_pass` field, and `full_encoder_precision_accepted` is hard-coded
`False`. Picking a tolerance is the root's decision with the gold in hand, not
a post-hoc convenience here.

## Prepared-graph decisions worth reviewing

1. **Scoped attention replacement, no global patch.** Only
   `candidate._scaled_dot_product_attention` is reassigned, on an isolated
   diagnostic module, to an explicit causal FP32-math helper (FP16 Q/K/V in →
   FP32 scores → FP32 softmax → FP32 P@V → one FP16 out). It mirrors the CPU
   baseline (`SDPBackend.MATH`) and carries **no `±50000` clamp**, which is a
   910A-only construct with no CPU analogue. `torch.nn.functional` is untouched
   and nothing is installed into production.
2. **Chunked query axis (`SDPA_QUERY_CHUNK = 512`).** Bounds the FP32 score
   buffer. Row-wise softmax is exactly decomposable, so the math is unchanged,
   but GEMM tiling differs from a single-shot FP32 matmul; the receipt says so
   instead of claiming bit-equality with CPU.
3. **Bounded-memory path only.** This driver implements no optimized matmul
   kernel; it is a diagnostic, not an optimization.
4. **TP8 deviation recorded, not hidden.** Row-parallel projections round their
   FP16 partial GEMM output *before* the FP32 all-reduce, and that collective
   cannot restore precision already rounded away. Status string in the receipt
   is `known deviation, not fixed, not waived`. The candidate separately
   promotes the FP16 vocabulary-embedding all-reduce to FP32.
5. **Leaf-operator context is attributed, not measured here.** The receipt
   records the root's Round80 leaf-control finding (native Half biased Linear
   is bit-exact equal to a no-bias Half GEMM plus a Half bias add, i.e. one
   extra output rounding; unbiased ≈9.7e-6 vs biased ≈3.6e-4 relative RMS on
   CPU; the FP32-output grouped-matmul workaround fails 161002) under
   `context_from_root_leaf_controls` with an explicit
   `NOT measured by this driver` attribution.

## Batch-shape audit correction

The initial fresh-fork review claimed `[0]` in `_encode` was a guaranteed
shape bug. The root rejected that claim after following the actual tensor
shapes: input IDs are `(1,seq)`,vocabulary embedding yields `(1,seq,hidden)`,
and text attention/MLP/residuals preserve the leading batch. Returning a bare
tensor does **not** remove the batch. Therefore `_encode`'s `[0]` correctly
removes the singleton batch and is not a defect proof.

The corrected driver keeps the batch in its explicit diagnostic path for
hooks,then probes the real private-candidate `encode_ids`. If that call matches
the diagnostic output,it uses `encode_ids` for warmed timing,including final
D2H. An observed exception is recorded without attributing it to an assumed
shape defect. No production method was edited.

## How the root is expected to run it

```bash
# root prepares the private root and stages the pieces:
#   /tmp/h3_round80_full_fp16_01/
#     candidate_encoder79.py        (SHA 2141b29a…)
#     evidence_io80.py              (EvidenceWriter)
#     encode_fp16_tp8_80.py         (this driver)
# environment (per rank, via torchrun):
#   VLLM_OMNI_910A_TE_DTYPE=half
#   VLLM_OMNI_910A_TEXT_FP32_LINEAR=0
#   VLLM_OMNI_910A_TEXT_PAD_KV=0
#   VLLM_OMNI_910A_VISION_FP32=0
#   TORCH_DEVICE_BACKEND_AUTOLOAD=1 (torch_npu must load)
torchrun --nproc_per_node=8 encode_fp16_tp8_80.py
```

Optional cheap wiring check before the 51.5 GB load:

```bash
H3_R80_PLAN_ONLY=1 torchrun --nproc_per_node=8 encode_fp16_tp8_80.py
```

Plan-only still enforces the CPU gold gate and fixture/checkpoint hashes, then
returns **before** any distributed or model setup. It writes
`plan_rank<rank>.json`, a separate filename, so it cannot consume or block the
real `tp8_rank<rank>.json` write-once receipts.

The driver fails closed if: the CPU gold is not complete **and**
repeat-verified; the CPU source/parameter SHA changed; any CPU stage payload
hash changed on disk (including the 12 used checkpoint-shard LFS hashes via the
Round78 manifest); the candidate SHA changed; the fixture or checkpoint config
changed; the graph is not uniformly FP16; any vision parameter conversion hash
mismatches the CPU reference; the NPU output is non-finite; a repeat forward
differs; or parameters/inputs change mid-run.

The root takes the **rank maximum** across the eight receipts for the timing
number; this driver never collapses ranks.

## Explicit non-claims

- The actual TP8 first pass completed all50 layers onall8ranks,but the second
  pass failed the repeat-bit check on every rank. See [failure summary](<failed_summary80.json>)
  and [original run log](<run.log>). CPUFP16 selected50 relativeRMS is7.5698%,
  repeat relativeRMS0.9011%;no warmed encode latency was measured after failure.
- The source is a private candidate,not a production change or49-step generation
  run. No official-lossless or complete-encoder2s acceptance is claimed.
- Round79's vision-only1.4568% result is a separate stage measurement;it cannot
  substitute for the full50-layer7.5698% comparison. Neither is waived.