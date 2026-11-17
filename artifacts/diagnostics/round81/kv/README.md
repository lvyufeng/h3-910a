# Round81 — FP16 K/V small-N zero-padding adapter (source only, **NOT EXECUTED**)

Scope: this directory only. The adapter patches **one private Round79 candidate
class**, never production code, never the CPU FP16 gold, never the Round80
driver.

**Status: prepared source only.** [`kv_padding81.py`](<kv_padding81.py>) was
never run on the NPU by its author. Nothing here is a precision result, a speed
result, or a claim that the encoder will be repeat-stable or finish under 2 s.

## The measured problem this addresses

Round80 ran the full TP8 FP16 candidate and got **repeat instability**: the
second forward differed from the first on all 8 ranks (0.90% relative RMS), so
the run failed closed before any warmed timing was taken.

Then a same-input micro control on 910A
([`round80/kv_padding.json`](../round80/kv_padding.json)) isolated one cause:

| Geometry `M=2043, K=5120` | Repeat bit-equal | CPU FP16 relative RMS |
|---|---|---|
| `N=128` (native K/V) | **0 / 8** | 1.05e-5 |
| `N=256` with zero rows | **8 / 8** | 1.46e-5 |
| `N=512` with zero rows | 8 / 8 | 1.46e-5 |
| `N=1024` with zero rows | 8 / 8 | 1.46e-5 |

Only **all-zero output rows** were appended and the extra columns discarded, so
the original FP16 weight values are untouched and the function is mathematically
unchanged — only the GEMM tiling moves. The CPU difference does change
(1.05e-5 → 1.46e-5), so this is a *different* tiling, not a pseudo-exact trick.

## What the adapter does

`install(candidate)` reads `MiniMaxH3Qwen3VLTextAttention.forward` with
`inspect.getsource`, rewrites it with an AST pass, compiles the result, and
assigns it back to that one class.

Exactly **two** call sites change:

```python
query = F.linear(hidden_states, q_weight)        # untouched
key   = _r81_kv_linear(hidden_states, k_weight)  # rewritten
value = _r81_kv_linear(hidden_states, v_weight)  # rewritten
```

The transform is fail-closed: if it does not match exactly `k_weight` and
`v_weight` (no more, no fewer), it raises `KVPaddingError` rather than installing
a partially rewritten graph. The Q projection and both MLP projections are
asserted untouched.

`_r81_kv_linear(x, weight)` pads **only** when all of these hold:

- `weight.shape[0] == 128` (the K/V width),
- `weight.dtype == torch.float16` and `x.dtype == torch.float16`,
- the tensor lives on an NPU/private-use device.

Otherwise it falls through to the original `torch.nn.functional.linear(x, weight)`
unchanged — no BF16 path, no FP32 weight shadow, no changed behaviour anywhere
else.

### Cache semantics

- The padded tensor is a **fresh, zero-offset** `(256, 5120)` FP16 buffer built
  with `torch.zeros` + a `detach()`ed copy of the real prefix.
- It is verified once at fill time: prefix difference must be exactly `0.0`,
  trailing rows must be exactly zero, storage offset must be `0`, and the entry
  records `requires_grad=False` / `is_leaf=True` so no autograd graph is held.
- The cache key is the real parameter identity —
  `untyped_storage().data_ptr()`, `storage_offset()`, `shape`, `stride`, `dtype`,
  `device`, `_version`. If a known storage changes shape/stride/version, the
  adapter **raises** instead of silently re-caching a mutated weight.
- Cost: 100 entries (50 layers × K,V) × 256 × 5120 × 2 bytes ≈ 262 MB. That is
  real device memory and is reported in `snapshot()['cache_bytes']`.

## API

```python
from kv_padding81 import install

kv_adapter = install(candidate)          # patch; returns KVPaddingAdapter
kv_adapter.prime(encoder.text_model)     # optional; fill all 100 caches untimed
...
r['kv_padding'] = kv_adapter.snapshot()  # measurement dict, no tensors
...
kv_adapter.uninstall()                   # finally: restore forward + clear cache
```

- `install(candidate, pad_rows=256)` — `pad_rows` is declarable but any value
  other than 256 is an unvalidated experiment.
- `prime(text_model)` — eager fill under `inference_mode`/`no_grad`; returns the
  number of entries built. Call it after the encoder is moved to the NPU and
  after the driver's excluded first call, so cache construction is never inside a
  timed window.
- `snapshot()` — `cache_entries` / `expected_cache_entries` / `cache_bytes` /
  `padding_bytes` / `max_abs_prefix_mismatch` / `tail_nonzero_count` /
  `all_cached_padded_are_zero_offset` / `no_cached_padded_requires_grad` /
  `helper_calls` / `padded_calls` / `passthrough_calls` / `cache_hits` /
  `transform` (source hashes and matched call names) and the permanent flags
  `not_executed_here: True`, `proves_full_encoder_precision: False`.
- `clear()` — drop the cache; safe before offload.
- `uninstall()` — restore the original `forward` and clear.

## What this cannot show

- Micro-scale `N=256` repeat stability does **not** prove the 50-layer graph is
  repeat-stable, accurate against the CPU FP16 gold, or faster.
- It says nothing about the encoder ≤ 2 s target, DiT/VAE, or official-lossless
  comparison.
- Round80's full-run 7.57% CPU difference and 0.90% repeat divergence are not
  waived by this adapter; they are the thing being tested.

## Verification status

[`test_kv_padding81.py`](<test_kv_padding81.py>) is standard-library only: it
parses the real candidate source as text (`ast.get_source_segment`) and never
imports torch, so the adapter module must stay importable without torch — every
torch use is a lazy import inside a function.

[`manifest81.json`](<manifest81.json>) pins the adapter SHA, the immutable
Round79 candidate SHA `2141b29a…`, the original/transformed forward source
SHA256, the expected matched call names, and the Round78 CPU gold receipt.