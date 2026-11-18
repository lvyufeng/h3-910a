# Round91 — bit-exact geometry cache for the vision tower (source only)

SOURCE ONLY. The author of this module has not executed it against torch, torch_npu,
an NPU, a model, the R83 launcher or CANN: no SSH, no NPU/GPU work, no model run, no
kernel build, no installation, no production edit and no commit. The real numbers
(primed-tensor equality, full-vision output SHA, timings, 8-rank repeats) belong to
the parent root driver that primes the cache inside the real encoder run.

## What it patches, and why it is lossless

`MiniMaxH3Qwen3VLVisionModel` recomputes two purely geometric tensors on every full
vision forward:

| Operator | Real output | Inputs it depends on |
|---|---|---|
| `rot_pos_emb(grid_thw)` | Float32 `[8064, 36]` | grid values, `rotary_pos_emb._inv_freq` (CPU float32, 18 entries), `dim`, `theta`, `spatial_merge_size` |
| `fast_pos_embed_interpolate(grid_thw)` | Half `[8064, 1152]` | grid values, `pos_embed.weight` (Half), `num_grid_per_side`, `spatial_merge_size` |

Both are independent of pixels. `prime()` calls the **captured original** operators
twice, requires raw **byte** identity of both outputs (`torch.equal`-strength, not a
numeric tolerance), checks that neither the parameters nor the input grid moved, and
stores deduplicated clones with their raw SHA-256. The enabled wrappers return those
already-computed tensors, so no arithmetic, cast, layout or mantissa changes: it is
the original result computed early. No TIK kernel and no float weight shadow is
involved, and the CPU float32 `_inv_freq` table keeps its original dtype.

Nothing else is cached: no pixel values, patch embeddings, hidden states, DeepStack
outputs or text conditioning. Pixels, all 27 vision blocks and all 50 text layers
still execute inside the parent's real encoder warm.

## Exact API

```python
import geometry_cache91 as gc

cache = gc.VisionGeometryCache91(vision)          # installs; DISABLED
cache.prime(grid)                                 # NPU int64 (2,3) real fixture grid
cache.set_enabled(True)                           # enable only after a successful prime
snapshot = cache.snapshot()                       # JSON-able evidence dict
cache.uninstall()                                 # pristine restore, idempotent
```

Convenience: `gc.install(vision)` and `gc.prime_and_enable(vision, grid)`.

## Fail-closed contract

Every enabled call re-validates the structure captured at prime time and raises
`GeometryCacheError` on any drift instead of returning a stale tensor:

* `pos_embed.weight` full tensor signature (pointer, storage offset, shape, stride,
  dtype, device, `_version`) — an in-place write bumps `_version` in torch;
  `pos_embed.weight` stays Half on the NPU, `_inv_freq` stays CPU float32;
* `_inv_freq` signature **and** its raw value SHA, `dim`, `theta`,
  `num_grid_per_side`, `spatial_merge_size`, and the configured vision geometry;
* the grid key is validated **by value**: the fast path is used only when the caller
  passes the very same tensor object whose signature still matches; any other object
  (or a version bump) has its values re-read and compared. A recycled `data_ptr`
  therefore can never serve a different grid, and the cache only ever serves the
  recorded real fixture grid (`REAL_GRID`, raw int64 SHA pinned);
* `prime()` is refused while enabled, so re-priming is explicit rather than a silent
  rebind; `prime()` also verifies the input grid was not modified.

## Verification performed here (stdlib only)

28 stdlib tests cover: exactly two per-instance method patches (class and global torch
untouched), pristine restore/identity guard/idempotence, disabled path = exact
original, enable-before-prime refusal, double-call raw-bit identity at prime, cached
shapes/dtypes/`extra_bytes` (18,579,456 + 1,161,216 B), returned cached object
identity with no recomputation, new-identical-grid acceptance by value, different-grid
refusal, version-bump re-read, recycled-pointer safety, changed weight/`_inv_freq`/
`theta`/`num_grid_per_side`/config refusal, grid-byte pin, the recorded 48-byte
fixture hash, non-claims, cached-tensor mutation detection and the reviewed source
pins (candidate `2141b29a…` plus both method source digests).

These tests use duck-typed stdlib doubles; they are a structural/semantic proof, not
an accuracy or speed result.

## What the parent must still measure

Prime on the real NPU grid, then compare the **complete** real vision (and full TP8
encoder) output SHA with the cache disabled, across all 8 ranks, including a freshly
allocated grid tensor and the enable/disable bit-recovery toggle. Only that evidence
can show losslessness on the real graph; this module must not be enabled by default
before those receipts exist.