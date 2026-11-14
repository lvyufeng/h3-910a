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
