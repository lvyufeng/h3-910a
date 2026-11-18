# Round92 — original HF CPU FP16 layer-25 same-input reference (source only)

> **Root execution note:** the original default CLI reached the reference-gate
> calculation but its local optional `tensor_info=None` shadowed the callable.
> It produced no reference/output; [the failure](<../results/cpu01/CPU_block25_failure92.json>)
> is retained and the executed source stays immutable. Use the
> [explicit-metric wrapper](<../run_cpu_reference_fixed92.py>) for the tested path.
> [Root results](<../README.md>) reproduce the R78 raw25 SHA twice and report the
> actual CPU/NPU same-input comparisons. The source-only wording below describes
> the fresh author's preparation, not the later root execution.

**Status: SOURCE ONLY.** Nothing in this directory has been executed by its author
against torch, transformers, a CPU model, an NPU or CANN: no SSH, no install, no
production edit, no commit. There is **no accuracy result, no speed result and no
lossless-pass claim** here. The numbers this program will print belong to the root
agent that stages it.

## Why one CPU layer is the next measurement

Round91 measured the real eight-rank FP16 encoder end to end and found that the
leaf-corrected vision path, although better on vision leaves, made the *whole*
encoder worse (7.42% → 17.04%), while early/mid text layers improved and layer 49
amplified. That leaves two different explanations that Round91 could not separate:

1. the NPU block arithmetic itself drifts, or
2. the NPU block gets *different inputs*, and layer 25 amplifies that difference.

Evaluating the **original HF CPU FP16 layer 25** on the actual NPU raw layer-24
state is an input-only sensitivity control that separates those cases. It is not a
Float-weight teacher, it casts no BF16 gold output, and it chooses no tolerance.

## The reference gate

`CPU_raw24` is the frozen Round78 raw layer-24 tensor
(`FP16_layer24.pt`, SHA-256 `6d08…`, tensor `7a21…`). Feeding it back through the
reconstructed original layer 25 must reproduce the frozen Round78 raw layer-25
output SHA-256 `a83f92f8…`, bit-exactly, twice. That gate is the whole point of the
module: it validates the pinned original HF source (`fcb557…`), the original BF16 →
FP16 conversion of the eleven layer-25 tensors (each checked against
`parameter_conversion[…]` of the Round78 receipt), the M-RoPE rows, the
`create_causal_mask` call and `SDPBackend.MATH` attention. If the gate fails the
module raises and **no NPU input is read**.

## Inputs, exact

* Weights: official dense checkpoint (`config.json` `d2dd…`, index `06c9…`). Only
  `model.language_model.layers.25.*` is read (11 tensors), converted BF16 → FP16 and
  SHA-verified against the Round78 conversion record. No Float32 linear weights.
* Positions: the frozen Round78 `FP16_multimodal_inputs.pt` (`6ab3…`) position rows
  (`05a9…`, shape `(3, 1, 2043)`, int64), split exactly as
  `Qwen3VLTextModel.forward` splits them; this build takes the `ndim == 3, rows != 4`
  branch, so `text_position_ids` is `None` and rotary uses all three rows.
* Causal mask: the original `create_causal_mask(config, inputs_embeds,
  attention_mask, past_key_values=None, position_ids)` call, with the original
  all-ones `(1, 2043)` mask and the original text `inputs_embeds` (`aed7…`).
* Rotary: `Qwen3VLTextRotaryEmbedding(text_config, device='cpu')`, Float32 CPU
  inverse frequencies, M-RoPE interleaved, FP16 cos/sin as in the original.
* Boundary: raw decoder layer 25 output, **before** DeepStack injection and
  **before** the final norm. No final norm and no DeepStack are added.

Weights, rotary and the causal mask are built once and shared by every case, so
only the layer-25 input tensor differs between cases — that is what makes this a
clean sensitivity control.

## API and artifacts

```python
from original_cpu_block25_92 import run_reference
receipt = run_reference(root=Path('/tmp/h3_round92_case01'), inputs_path=None)
# writes  <root>/CPU_block25_reference92.json
#         <root>/CPU_block25_outputs92.pt
# optional: compare=/tensor_info= from the staged full_row_fp32_fixed84 helper
```

CLI: `python original_cpu_block25_92.py --root /tmp/h3_round92_case01`.

The root supplies `<root>/NPU_raw24_inputs92.pt`:

```python
{'inputs': {'native': half(1, 2043, 5120), 'corrected_vision_native_text': ..., ...},
 'position_ids': original (3, 1, 2043) int64 rows,
 'source_raw24_SHAs': {case: actual raw24 SHA-256},
 'captured_source_SHA256': <root capture driver SHA-256>}
```

`CPU_block25_outputs92.pt` holds exactly

```python
{'outputs': {'CPU_raw24': ..., 'native': ..., 'corrected_vision_native_text': ..., ...},
 'input_SHAs': {same keys: actual input SHA-256},
 'position_ids': original CPU positions}
```

Case names are restricted to the five Round91 case names whose actual raw layer-24
outputs are pinned from the committed Round91 receipt; a declared SHA that disagrees
with the computed tensor or with the Round91 receipt is refused. When the root also
leaves its own `expected_raw24_92.json` next to the bundle, that file is cross-checked
too: its pins must agree with the committed Round91 receipt *and* with the bundle's
declared SHAs, or the module refuses. `run_reference`
never overwrites an existing reference or outputs file, and it records the output
file SHA-256, the inputs path/SHA-256 and the metric-helper provenance (staged
`full_row_fp32_fixed84` or an identical internal fallback).

The file exceeds the soft 400-line preference because the fail-closed validation
ladder (receipt gates, per-tensor conversion SHAs, position/mask provenance, repeat
and frozen-state invariants) is retained as instructed.

## What passing would and would not mean

A completed run reports, per case, CPU block 25 on the actual NPU raw24 versus CPU
block 25 on the frozen CPU raw24: input relative RMS, output relative RMS and their
ratio. That bounds how much of the block-25 gap is input-driven. It **cannot** by
itself prove a unique hardware operator bug, a lossless pass, whole-encoder FP16
agreement, or any speed target; the set-point and acceptance call remain with the
root, against the unchanged CPU FP16 reference.

## Contract tests

`test_original_cpu_block25_92.py` runs without torch: it imports the module as a
stdlib-only module, parses its AST, and cross-checks every pinned value against the
committed Round78 receipt, the committed Round91 actual receipt and the committed
pinned original HF source (including that the `create_causal_mask` keyword list and
the three M-RoPE split branches match byte-for-byte in structure).