# Round95 — original-HF CPU Half text M-RoPE table cache (source only)

**Status: SOURCE ONLY.** Nothing here has been executed by its author against
torch, torch_npu, an NPU, the HF model or CANN: no SSH, no model run, no kernel
build, no installation, no production edit, no commit. There is **no accuracy
result, no speed result and no lossless claim** in this directory. The numbers
belong to the root agent that stages this module and primes the cache.

## What this is

[`CPUTextRopeCache95`](cpu_text_rope_cache95.py) is a per-instance wrapper around
the reviewed `MiniMaxH3Qwen3VLTextRotaryEmbedding` from
`candidate_encoder79.py` (SHA `2141b29a…`). `prime(x, position_ids)` computes the
**original HuggingFace CPU** `Qwen3VLTextRotaryEmbedding` table for the pinned
real M-RoPE positions:

- `x` is used **only** for `dtype`/`device` metadata. The original HF forward
  consults `x.device` and `x.dtype` only, so the CPU table is computed on an
  empty `torch.float16` CPU tensor. No actual hidden state is copied to CPU,
  cached, or used as a teacher.
- `position_ids` must be the pinned real R78 fixture: int64 `(3,1,2043)` on the
  NPU whose values hash to `05a92359…`. The table is computed twice and must be
  raw-byte identical, then the original CPU cos/sin are copied **bit-exactly**
  to the NPU and must reproduce the R94-measured original CPU table
  (`cos f0480c87…`, `sin 0c147aa3…`).
- `inv_freq` is pinned (`29fc897b…`) and the candidate `_get_inv_freq(cpu)` must
  be raw-byte equal to the original HF CPU inverse frequencies. The absolute
  frequency math is already correct; this cache does not re-derive it.

The cached table is `cos`/`sin` Half `(1,2043,128)`: **1,046,016 B per rank**. It
is a RoPE table only — no pixel, patch, hidden, DeepStack or conditioning value.

## This is not a lossless native-bit cache

Round94 measured the native R79 NPU Half text cos/sin against this exact
original CPU table: cos relative RMS `7.1403e-6` (max exactly one Half ULP
`0.00048828125`), sin `2.80e-8` — **not bit-equal**. Serving the CPU table is a
*precision alignment* to the original HF reference, not a bit-preserving replica
of the native NPU rounding. `snapshot()` states `native_rounding_changed: True`,
the measured `native_equals_CPU_table_bits`, and
`full_encoder_precision_accepted: False`, `numerical_tolerance: None`. Whether
this improves the complete encoder is a separate root measurement; Round91 showed
better leaves can make the whole graph worse, so no gain is assumed here.

## Fail-closed behaviour

- Provenance: candidate file SHA, reviewed class name/module, and line-based
  source digests of `__init__`, `_get_inv_freq`, `forward`; original HF modeling
  source SHA.
- Config: `dim == 128`, `mrope_section == [24,20,20]`, default `rope_type`,
  `attention_scaling == 1.0`, zero learnable parameters on both rotaries.
- Prime inputs: pinned x metadata, pinned position fixture, pinned CPU table.
- Every enabled call: config fingerprint, candidate/HF source SHA, x
  shape/dtype/device identity (including device index), position metadata and
  (when a version counter is unavailable) values, and cached tensor
  signature/version. A recycled `data_ptr`, a new object or an inference tensor
  never takes the fast path; a mutated cached tensor raises.
- `uninstall()` restores the instance `forward` only when it is still ours; a
  third-party patch is refused. Class level, `torch.nn.functional` and private
  candidate functions are never touched. Re-priming while enabled is refused.

## Root integration notes

1. Stage `candidate_encoder79.py` beside this module (or pass `candidate_path=`).
2. After the encoder weights are loaded and moved to `npu:<rank>`:
   `cache = CPUTextRopeCache95(encoder.text_model.rotary_emb, candidate_path=...)`.
   The constructor installs but leaves it **disabled** (native path).
3. Move the R78 `position_ids` fixture to the rank device (int64,
   `(3,1,2043)`), then `cache.prime(x, positions)` **outside the warm clock**.
   `x` is any tensor with the fixture metadata `(1,2043,5120)`, Half, on that
   NPU device; its content is never read. Pass the R84 `compare`/`tensor_info`
   helpers to record native-vs-CPU metrics consistently with other receipts.
4. `cache.set_enabled(True)` per case; `set_enabled(False)` to return to native;
   `cache.uninstall()` at the end. `snapshot()` re-hashes the cached tensors.
5. In the real encoder, `position_ids` are rebuilt each forward inside
   `inference_mode`, so they carry no version counter: the cache value-validates
   the 49 KB position tensor on every call. That is the intended fail-closed
   path, not a bug.
6. Expected extra device memory: **1,046,016 B/rank** (`extra_bytes` in the
   snapshot). This is not conditioning storage.
7. Do not enable this by default for production; it is a precision-alignment
   candidate for the root's paired full-TP8 A/B cases
   (native vs CPU table, native/precise text, vision native/corrected, with the
   R91 bit-exact geometry cache kept always on). The R91 combined vision
   path that measured 17% whole-text error is not made acceptable by this table.

## Files

- [`cpu_text_rope_cache95.py`](cpu_text_rope_cache95.py) — the cache module.
- [`test_cpu_text_rope_cache95.py`](test_cpu_text_rope_cache95.py) — stdlib-only
  contract tests (duck-typed doubles; no torch/NPU/HF execution).
- [`manifest95.json`](manifest95.json) — hashes, pins, API, non-claims.