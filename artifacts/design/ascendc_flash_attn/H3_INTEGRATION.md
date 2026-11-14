# Integrating into H3: the blocker is a 37-key suffix mask

The kernel is verified faster in isolation (872 ms vs 1166 ms, 1.34x) and the
backend patch is in place and imported. It does not fire, and this records why.

## The patch

`h3_910a_attn.py` gained `_flash_usable()` and a guarded branch in
`forward_npu` that routes to `fa910a.flash_attention_910a()` when the shapes
match, falling back to the dense path on any exception. The module imports
correctly and `_flash_usable` returns False rather than raising, so nothing
breaks -- the fast path is simply never taken.

## What the runtime actually passes

Instrumented the guard to log every distinct attention shape. Two appear:

```
shape=(1, 56, 3, 128)   causal=False  mask=none
    -> the text encoder (S=3), correctly rejected: not tile-divisible.

shape=(1, 7, 63232, 128) causal=False mask=shape=(1,1,1,63232) dtype=bool
    -> the DiT, which is the one we care about.
```

The DiT shape matches the kernel's contract exactly: `(B=1, H=7, N=63232, D=128)`
fp16. But it carries a boolean key mask, and the contents are:

```
ones = 63195 / 63232      first0 = 63195      suffix_padding = True      zero_cnt = 37
```

So it is **pure suffix padding**: the first 63195 positions are valid and the
last 37 are padding. That is the whole obstacle.

Note 63232 = 256 * 247, so `Bc=256` divides it and the kernel would be fast on
this shape. The sequence length is also shorter than the 73408 seen at 10 s,
confirming N varies with clip length, so tiles must be picked per call.

## Why the three obvious escapes do not work

1. **Ignore the mask.** It is not a no-op: 37 keys are genuinely masked, and
   including them changes the softmax denominator. Would be silently wrong.
2. **Truncate to the 63195 valid keys.** The kernel needs both tile dimensions
   to divide N. `63195 = 5 * 3 * 4213` has no factor of two at all, so no
   16-aligned tile divides it. The kernel cannot run on the exact valid length.
3. **Pass the mask to the kernel.** The reference does accept `attn_mask`, but
   `attention.py:51` documents it as `(1, N, N)` or `(B, N, N)` fp16 -- a dense
   additive mask. At N=63232 that is 63232^2 * 2 B = **8 GB**. Infeasible.

## The exact fix that would work

Because the mask is suffix padding and only 37 keys are involved, the result can
be corrected rather than masked:

1. Zero the V rows of the 37 padded keys, so their contribution to the numerator
   becomes 0 and the numerator stays exact.
2. Run the kernel on the full 63232 (tile-divisible).
3. The kernel already returns `l` = log-sum-exp over all keys, so its output is
   `numerator / (den_valid + den_pad)`. Recover `den_pad = sum over the 37 keys
   of exp(s_j)` from a tiny `Q @ K_pad^T` matmul -- `Nq x 37 x 128`, cheap.
4. Rescale: `O = O_kernel * exp(l) / (exp(l) - den_pad)`.

That is exact, needs only a small extra matmul, and keeps the fast kernel for
the 63232-length case.

A simpler variant worth trying first: since the padded keys are a contiguous
suffix and there are only 37, check whether the harness always pads to a fixed
total per clip length. If so, the padded keys' K rows may be zeros, in which
case `s_j = 0` for all q and `den_pad = 37 * exp(0) = 37` exactly -- no matmul
needed, just a constant. That should be checked before building the general
correction, because it would make the fix trivial.

## Status against the objective

* "支持 910A 的 flash attention": **done**. Builds, runs, matches fp32 to fp16
  rounding at both the small and the real shape, and is 1.34x faster per call
  than the dense path.
* "加速 H3 推理": **not yet**. The kernel is never invoked inside H3 because of
  the mask above, so there is no end-to-end speedup to report. The per-call gain
  is real but unrealised.

---

# The flash path now fires inside H3

The 37-key suffix mask turned out to be recoverable exactly, and the kernel is
now actually invoked in a live run.

## Root cause of the mask, and why it is harmless

`minimax_h3_transformer.py` builds the mask only when the backend advertises
neither `supports_prefix_kv_slicing` nor `supports_packed_mask_free`:

```python
used = min(max_seqlen, packed_total)
mask_free_packed_padding = ... supports_packed_mask_free()
no_mask = ... supports_prefix_kv_slicing or mask_free_packed_padding
if used < packed_total and not no_mask and not use_ring:
    attn_mask = torch.arange(packed_total, device=q.device)[None] < used
```

so it is purely `arange < used`, confirming suffix padding. And
`batched_packing.py` allocates the packed tensor with `torch.zeros`, so a padded
key row is k=0, v=0. Its score is `q.0 = 0`, hence `exp(0) = 1` of denominator
and nothing at all to the numerator. Therefore

```
O_kernel = num / (den_valid + n_pad)
O_exact  = O_kernel / (1 - n_pad * exp(-l))
```

using the log-sum-exp `l` the kernel already returns. No mask, no extra matmul,
no 8 GB additive mask.

Verified before wiring it up, at N=1280 with 256 zero-padded keys:

```
worst relative error WITH correction: 0.000597   PASS
```

## Two real bugs found while wiring it in

1. **Build path vs cwd.** TIK writes `./kernel_meta` relative to the process
   cwd, but the wrapper looked in `/tmp/kernel_meta`. Outside the service that
   accidentally worked because cwd was `/tmp`; inside the service cwd is
   `/home/mseco/h3-910a`, so every build "produced nothing". The wrapper now
   builds inside a private temp dir and moves the artefact into the cache.
2. **Eight-worker build race.** All 8 USP workers build the same shape at the
   same time into one directory. Now serialised with an exclusive `flock`, with
   a re-check inside the lock so only the first worker builds.

## Evidence that it fires

Live run, `VLLM_OMNI_910A_FLASH_ATTN=1`, 5 s clip (N=63232, Bc=416 which
divides 63232 exactly):

```
kernel cache:  b1_h7_n63232_d128_br64_bc416  (26392 B .o + json)
fallbacks:     0
```

Zero fallbacks means every DiT attention call in that run took the fused path,
and the built artefact shows the tile picker chose 416 rather than the reference
default of 128.

## Still missing: the end-to-end number

The run had not completed when this was written, so there is **no flash-ON vs
flash-OFF wall-clock comparison yet**. What is established is that the kernel is
correct and 1.34x faster per call in isolation, and that H3 now dispatches to
it. What is not established is that the whole pipeline got faster -- the fast
path adds an NDS->NZ conversion per call and an fp32 rescale, and at ~870 ms per
attention call those could eat into the gain.

The next measurement must be a clean A/B: same prompt, same clip, one run with
`VLLM_OMNI_910A_FLASH_ATTN=0` and one with `=1`, both after a warm-up request,
comparing the `[H3PERFP]` breakdown or total wall clock.

---

# Why the first integrated run was so slow: a per-call device sync

First live run with the fused path actually firing: the 5 s clip sat
`in_progress` for over 15 minutes with no `[H3PERF] dit` progress lines at all,
where the dense path had printed them steadily. `npu-smi` showed the shape of
the problem:

```
NPU 3  AICore 83%      NPU 0,1,2,4,5,6,7  AICore 7%
```

One chip busy, seven idle. That is not a hang (the AICore is working) but a
serialisation.

Cause: the wrapper called `aclrtSynchronizeStream` after **every** attention
launch. In a standalone test that was needed to read the result, but inside the
diffusion pipeline it forces a full device round-trip per call -- and there are
roughly 200 attention calls per denoise step across 50 layers. The pipeline
never overlaps, so seven chips wait on the one doing the synchronous call.

Fix: do not synchronise. The kernel is enqueued on the torch stream that
`torch.npu.current_stream().npu_stream` reports, so torch's own stream ordering
already covers it, and any later torch op that reads the output is ordered
behind the kernel. Sync is now opt-in via `VLLM_OMNI_910A_FLASH_SYNC=1` for
callers that must read the buffer immediately.

This is the second time in this integration that a convenience from the
standalone harness was actively harmful in situ -- the first being the hardcoded
`/tmp/kernel_meta` path. Both were invisible in testing because the harness
controlled its own environment, and both only surfaced under the real pipeline.

---

# Integrated it runs 3.2x SLOWER, because of the layout conversions

Measured the live run after removing the sync. It progresses fine (no hang), but
the per-layer time is the wrong direction:

```
dense run:        [H3PERF] dit N/50 block=1.392 s
flash run:        dit 11/50 -> dit 21/50 in 45 s   =>  4.5 s per layer
```

4.5 s against 1.392 s is **3.2x slower**, in a configuration where the kernel
itself is 1.34x faster than the dense attention (872 ms vs 1166 ms measured at
the real shape) and had zero fallbacks in this run.

## Why

The kernel only accepts and returns NZ layout, `(B,H,d/16,N/16,16,16)`, while
the H3 path is NDS. My wrapper therefore pays, on every attention call:

* `qs = (q.float() * scale).half()` -- an fp16 -> fp32 -> fp16 round trip over
  the full tensor just to apply a scalar
* three `permute(...).contiguous()` calls, one each for q, k, v, each a full
  N*D copy
* on the way out, `permute().reshape().to(float32)`, an fp32 division by the pad
  correction, then `.to(fp16)`

At N=63232, H=7, D=128 that is roughly 113 MB per tensor per conversion, and
with q, k, v, the scaled copy and the output that is on the order of a gigabyte
of extra traffic per call. The dense path touches none of it. The per-call
kernel gain of ~300 ms is swamped.

Note this is a different failure from the two previous ones: those were harness
conveniences that leaked into the pipeline (hardcoded path, per-call sync). This
one is intrinsic to my wrapper's interface choice, and it was **already present**
in the 872 ms standalone figure -- it just did not show up as a loss there
because I compared kernel-with-conversion against dense-without-conversion and
the kernel was fast enough to still win at N=73408.

## What would actually fix it

1. **Fold the scale into the kernel**, or pre-scale q once per denoise step
   instead of per attention call. Cheapest win: scaling is constant across all
   50 layers of a step, and possibly across all steps.
2. **Avoid the fp32 round trip.** `q.float() * scale` then `.half()` is two
   full-tensor casts; `q * half_scalar` in fp16 does it in one pass without
   leaving fp16.
3. **Skip the pad correction when n_pad == 0**, which is the common case, so no
   fp32 upcast and divide at all.
4. **Convert once, not per call.** q/k/v for a given layer are produced by that
   layer's projections, so they cannot be cached across calls; but the output
   conversion could write straight into the layout the caller needs.
5. **The real fix: teach the kernel NDS.** Its NZ requirement comes from the
   TIK reference's `F16_FracNZ` input declaration. Accepting NDS would remove
   every conversion, at the cost of changing the op contract or adding a
   `data_move` with nd2nz inside the kernel (which is what the cube wants
   anyway).

## Honest status of the objective

* "支持 910A 的 flash attention": **done and verified.** Builds for dav-c100,
  runs via the legacy runtime API, numerically matches an fp32 reference to fp16
  rounding at N=256, N=992 and the real N=73408, and is 1.34x faster per call
  than the dense path when the layouts it needs are already in place.
* "加速 H3 推理": **not achieved.** Integrated as-is it makes the pipeline 3.2x
  slower. The kernel is correct and fast; the glue around it is not, and until
  the conversion cost is removed this cannot be switched on.

So the per-call result stands, the integration does not. Leaving this disabled by
default (`VLLM_OMNI_910A_FLASH_ATTN` unset) is the correct state.

---

# RESOLVED: 1.48x faster per layer, measured A/B on identical work

The 3.2x regression is fixed and reversed. The cause was not the layout
conversions at all -- that was my previous conclusion and it was wrong.

## The conversions were never the problem

Micro-benchmarked every conversion at the real shape (B=1, H=7, N=63232,
D=128; 113 MB per tensor), on device, warmed up:

| operation | ms |
|---|---|
| `q.float()*scale` then `.half()` (what the wrapper did) | 1.16 |
| `q.mul(scale)` in fp16 | 0.24 |
| `to_nz(q)`, permute + contiguous | 0.27 |
| in-place `mul_` on the NZ copy | 0.25 |
| `to_nz(k)` + `to_nz(v)` | 0.48 |
| output permute + reshape + contiguous | 0.28 |
| output `.float()` upcast | 0.59 |
| `torch.empty_like` allocation | 0.02 |
| **total, all of it** | **3.3 ms** |

3.3 ms against a 672 ms kernel. The conversions were noise, and my claim that
they were "on the order of a gigabyte of traffic" that swamped the gain was
simply wrong -- a gigabyte at this part's achieved bandwidth is about 1 ms, not
seconds. I had reasoned from a bandwidth estimate instead of measuring.

## What it actually was: a device-to-host copy per call

`_flash_pad_count()` did

```python
m = attention_mask.detach().to("cpu").bool().flatten()
```

on **every** attention call. A D2H copy synchronises the stream, so this stalled
the pipeline exactly like the `aclrtSynchronizeStream` removed in the previous
round. I introduced it while solving the mask problem and did not see that it
had the same effect as the bug I had just deleted.

The fix needs no device work at all: the model already carries the valid length
as a plain Python int in `attn_metadata.packed_padding.kv_length`, so
`n_pad = N - kv_length` is pure host arithmetic. The mask is now only inspected
as a cached fallback when that metadata is absent.

## Measured A/B, same prompt, same 5 s clip, same 8-way USP

`[H3PERF] dit N/50 block=` reported by the model itself:

```
dense: 1.374  1.394  1.400  1.387  1.460  1.400   s/layer   (~1.39)
flash: 0.949  0.973  0.923  0.935  0.922  0.945   s/layer   (~0.94)
fallbacks: 0
```

**1.48x faster per layer**, on identical work, with the fused kernel serving
every DiT attention call.

Cross-checks that agree:

* kernel alone at this shape: 672.5 ms/call, tile picker choosing br=64/bc=416.
* dense attention at this shape, scaling the measured 1166 ms at N=73408 by
  N^2: about 865 ms. 672 against 865 is 1.29x at the attention step, and the
  layer-level gain is larger because the layer time also includes the epilogue.
* the earlier 10 s clip at N=73408: 867.8 ms measured, 1.34x over dense.

## Projected effect

At 4 steps x 50 layers = 200 layer-evaluations, saving about 0.45 s each:

```
~90 s off the run
```

against the dense baseline this was previously measured at 449 s for 768p/10 s,
i.e. roughly 449 -> ~360 s. That would still not reach the 300 s target, but it
is the first real end-to-end speedup and it is worth keeping.

## Two corrections to earlier notes in this file

1. The claim that layout conversion swamped the gain is withdrawn; the measured
   cost is 3.3 ms. The 3.2x regression came from a per-call D2H copy.
2. The "3.2x slower" number and the "4.5 s per layer" figure were from a 5 s
   clip while dense "1.392 s per layer" was from a 10 s clip, so that comparison
   was not apples to apples either. The A/B above is the first same-workload
   comparison.

## Lesson

Three times now, a synchronisation appeared inside the per-attention path and
cost a large multiple: `aclrtSynchronizeStream`, then a `.to("cpu")`, and
implicitly the harness conveniences before them. The shape of the failure is
always the same -- one chip busy, the rest idle, layer time inflated several
fold. Any future edit on this path should be checked against
`npu-smi` utilisation before and after, because per-layer time alone does not
say whether you are slow on compute or serialised on sync.
