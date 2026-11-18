# Round93/cpu — original CPU FP16 raw layers 25..49 suffix propagation control

**SOURCE ONLY.** Authored here and never executed by its author against torch,
transformers, a CPU model, an NPU or CANN: no SSH, no install, no model run, no
production edit, no commit. Every number this program reports belongs to the root
agent that stages and runs it with the frozen Round78 CPU gold plus the actual
Round91/Round92 eight-rank raw-layer-24 capture.

## Why this probe exists

Round92 evaluated one original CPU layer (25) on the actual corrected-vision NPU
raw layer-24 state (`e232f61b...`). On that state the CPU block-25 output moved
only `0.2530746% -> 0.2697414%` (`1.065857x`), i.e. layer 25 alone does **not**
explain the whole late-stack gap. The next question is whether that input state
stays bounded through the **whole remaining selected suffix** or whether a later
layer amplifies it.

This module answers exactly that, with no NPU arithmetic involved:

```
baseline : CPU_suffix_25..49( CPU_raw24 )        -> must reproduce R78 gold
changed  : CPU_suffix_25..49( NPU_raw24 e232 )   -> input-only propagation
```

## Gates (fail closed, in this order)

1. `original.load_r78_receipt` — pinned receipt SHA `44acac61...`, completed /
   repeat-verified / pre-final-norm-identity flags, pinned torch/transformers
   versions, pinned HF modeling source `fcb5571e...`, 50-layer inventory, official
   parameter SHA `a69676ba...`.
2. **Conversion inventory (fail fast, seconds not minutes):** the Round78
   `parameter_conversion` map must hold all 11 entries for every layer 25..49
   (275 tensors). A missing entry raises before the 25-layer build, so a partial
   inventory can never burn the whole CPU budget.
3. `original.load_cpu_raw24` — frozen `FP16_layer24.pt`, file `6d081a27...`,
   tensor `7a217819...`.
4. `original.load_original_text_inputs` — frozen M-RoPE `position_ids`
   `05a92359...` and original `inputs_embeds`.
5. `original.build_layer` for **each** of layers 25..49 — 11 original BF16
   checkpoint tensors per layer converted to FP16 and byte-compared to the Round78
   per-parameter conversion SHA (275 tensors), `strict=True, assign=True`, real CPU
   FP16 parameters only.
6. **Baseline gate:** `CPU_raw24` through raw layers 25..49 must reproduce **all
   25** Round78 raw-boundary SHA256 values bit-exactly, and raw49 must equal the
   frozen `FP16_selected50.pt` tensor `82944620...` (not the serialized raw49 file
   SHA `98302d42...`). This runs **before any NPU
   input is read**; a mismatch raises and no numbers are produced.
7. Only then the actual bundle is read (`original.load_inputs_bundle`): each case
   SHA must equal the declared and pinned Round91 receipt values, and the bundle
   positions must be bit-equal to the frozen original rows. The corrected case must
   be `e232f61b...`.
8. The changed branch runs **twice** and all 25 boundary SHAs must be identical
   between passes; every boundary must also be finite.
9. Invariants: combined 275-tensor parameter SHA, CPU raw24, R78 receipt, R78
   multimodal inputs, R78 selected-50 payload, input bundle, the reused helper and
   this source are re-hashed after the run and must be unchanged.

## Staging and running (root)

```bash
# 1) stage the pinned helper byte-exactly next to the driver
sha256sum original_cpu_block25_92.py   # must be 838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b
mkdir -p /tmp/h3_round93_suffix01
cp artifacts/diagnostics/round92/cpu/original_cpu_block25_92.py /tmp/h3_round93_suffix01/
cp artifacts/diagnostics/round93/cpu/suffix93.py                /tmp/h3_round93_suffix01/

# 2) run on the CPU node (16 threads, MKLDNN, SDPA MATH); no NPU needed
cd /tmp/h3_round93_suffix01
python3 suffix93.py --root /tmp/h3_round93_suffix01 \
  --inputs /tmp/h3_round92_case02/NPU_raw24_inputs92.pt
```

Optional flags: `--helper <path>` (verified SHA anyway), `--case` (default
`corrected_vision_native_text`), `--baseline-repeats 2` (two full baseline passes;
costs roughly one extra baseline pass, only use with a larger time limit).

**Budget.** 25 layers × ~1GB ≈ 24.4GB of FP16 weights stay resident (plus ~1GB
host overhead); peak well under 64GB. Baseline pass, two changed passes and the
two parameter-hash sweeps are four traversals of the suffix: budget ≈ 5400s and
write progress is updated after the baseline gate and after each changed pass, so
a timeout still leaves evidence in `CPU_suffix_progress93.json`. The default
`--baseline-repeats 1` is safe because the baseline is already gated bit-exactly
against 25 independent gold boundaries; `2` additionally proves CPU determinism.

## Outputs (all inside `--root`, never overwritten)

| file | content |
|---|---|
| `CPU_suffix_reference93.json` | full receipt: gates, per-boundary SHAs, per-boundary metrics, error growth, invariants, output file SHA |
| `CPU_suffix_outputs93.pt` | `baseline_selected50`, `corrected_selected50` (raw49 FP16 CPU), `source_input_SHA256`, per-tensor SHAs, `layer_boundaries` |
| `CPU_suffix_progress93.json` | incremental progress (baseline gate done, each changed pass done) |
| `CPU_suffix_failure93.json` | only on failure: stage, error, traceback, which files existed |

A root holding any of these four files is refused, so a failed run must be
retried in a fresh root; the failure record is preserved rather than overwritten.

## What the receipt reports

* `baseline_pass.matches_R78_all_25_boundaries` and the full 25-SHA table.
* `corrected_passes[*].boundaries[layer]` — output info, R78 truth SHA, whether it
  matched (it should **not** match: the input state differs), and metrics versus
  the CPU baseline boundary (finite, max_abs_error, RMSE, relative_rms).
* `selected50` — baseline vs R78 truth (must be bit-equal), corrected vs R78 truth,
  corrected vs CPU baseline.
* `output_error_growth.per_layer` — per-boundary relative RMS, ratio to the
  previous boundary, the largest boundary, and the total
  `selected50_over_input_amplification`. These are descriptive: the receipt
  carries `numerical_tolerance: null` and no pass/fail claim.

## What this cannot show

`CPU_suffix(NPU raw24)` versus `CPU_suffix(CPU raw24)` isolates **input**
propagation only. It contains no NPU arithmetic, so it cannot attribute a gap to a
specific 910A operator, cannot be a Float/BF16 teacher, casts no BF16 gold output,
adds no final norm and no DeepStack, and selects no tolerance. Combined with the
Round92 block-25 arithmetic controls, it separates "different late input" from
"different late arithmetic" — nothing more.

## Reuse contract

Every heavy operation is imported from the pinned Round92 module and never
re-implemented here: `load_r78_receipt`, `load_cpu_raw24`,
`load_original_text_inputs`, `build_layer`, `split_original_positions`,
`build_rotary`, `build_causal_mask`, `run_block25`, `load_inputs_bundle`,
`install_helpers`, `tensor_info`, `compare_pair`, `bytes_sha`, `materialise`,
`raw_tensor_sha`. `install_helpers()` is called with no arguments so the reviewed
`full_row_fp32_fixed84` metrics are used rather than caller-supplied shadows. The
helper's inherited `only_layer_25_weights_loaded` metadata is stale for a suffix
run and is removed, never republished; the receipt states
`only_suffix_layers_25_to_49_loaded`.

## Files

* `suffix93.py` — the driver (CLI + `run_reference` API). It is 558 lines, above
  the ~450-line soft target on purpose: every gate and receipt field is inline,
  no standard-library-free helper logic is duplicated, and no gate was dropped to
  meet a line count.
* `test_suffix93.py` — 27 stdlib-only contract tests (no torch, no model, no NPU):
  gate ordering, reuse, exclusive writes, entity pins, non-claims.
* `manifest93.json` — file SHAs, pins, API/CLI and the non-claim flags.

Run the tests with:

```bash
python3 -m unittest discover -s artifacts/diagnostics/round93/cpu -p 'test_*.py'
```