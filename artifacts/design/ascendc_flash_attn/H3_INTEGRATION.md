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

---

# Caveat on the 1.48x: the A/B is sound, the full-run total is not measured

Worth separating two claims of very different strength, because the notes above
could be read as making the stronger one.

## What is well established

The same-workload, interleaved A/B at a 5 s clip:

```
dense: 1.374 1.394 1.400 1.387 1.460 1.400  s/layer
flash: 0.949 0.973 0.923 0.935 0.922 0.945  s/layer     fallbacks 0
```

Both runs on the same host, same 8-way USP, same prompt and clip length, read
off the model's own `[H3PERF] dit N/50 block=` line, in a comparable state
(encode 38.9 s and 38.4 s respectively). That is a ~1.48x per-layer gain and it
is not an extrapolation.

Cross-checks agree: the kernel alone is 672.5 ms at N=63232 with the tile picker
choosing br=64/bc=416, versus an estimated 865 ms for dense attention at that
shape (scaling the measured 1166 ms at N=73408 by N^2), and 867.8 ms / 1.34x
measured directly at N=73408.

## What is NOT established

**No clean full-run wall-clock total.** Every attempt to time a whole 5 s clip to
completion was confounded:

* runs take far longer than the per-layer numbers imply, and none of the 5 s
  clips has been observed to reach `completed`;
* a later flash run showed `encode=125.22s` against 38-39 s in the A/B runs, so
  the first request after a restart is cold and the absolute layer times in such
  a run (1.04-1.13 s) are not comparable to the warm A/B figures (0.91-0.98 s);
* the video `content` endpoint returned 98 bytes for an incomplete request, so
  no visual check of a flash-produced clip has been done either.

So the honest statement is: **the attention layer is ~1.48x faster on identical
warm work**, which is the part this kernel controls. Whether the whole pipeline
lands near the projected ~360 s is unverified, and one warm full run with a
completed video is still owed.

## Practical state

The fast path is off by default (`VLLM_OMNI_910A_FLASH_ATTN` unset) and the host
is left running the dense configuration. To evaluate it, set the variable, submit
one request to warm up, then measure the second -- the same discipline the
earlier dense measurements needed.

One thing to fix before trusting a full-run number: the first request after a
restart pays a 125 s encode instead of 39 s, a 3x penalty on a fixed cost, which
alone can hide or manufacture a few tens of seconds of difference in a total.

---

# Clean cold-vs-cold dstep comparison: 1.41x on the denoise phase

Previous notes compared a cold dense `dstep 1` against a warm flash `dstep 10`,
which inflated the gain. Comparing like with like -- both runs freshly started,
both reading `dstep 1`, so both pay the same cold penalty:

| `dstep 1` | prep | **model** | update | total |
|---|---|---|---|---|
| dense | 8.58 s | **113.62 s** | 8.92 s | 131.1 s |
| flash | 8.65 s | **80.66 s** | 9.53 s | 98.8 s |

`model` is the denoise work: **113.62 -> 80.66 s, or 1.41x**. `prep` and `update`
are unchanged within noise (8.58 vs 8.65, 8.92 vs 9.53), which is exactly what a
change confined to the attention kernel should look like.

This agrees with the interleaved per-layer A/B from the previous round
(dense 1.374-1.460 vs flash 0.913-0.981 s/layer, ~1.48x) and with this run's own
`dit 11/50 block=0.926s`. Cold and warm both land near 1.4-1.5x, so the gain is
not an artefact of thermal or cache state.

Warm flash for reference: `dstep 2 prep=6.09 model=51.61 update=2.42`. Warm
dense was never captured because the earlier dense run was stopped inside its
first step, so the warm dense figure remains inferred (~73 s by applying the
1.41x ratio) rather than measured.

## Why no request ever completed in 906 s: it is not the attention

Every 5 s clip I have submitted returned `in_progress` at the 906 s timeout, in
both dense and flash configurations. The cause is now clear and has nothing to
do with this kernel:

```
dstep counter reached 32 across two requests   ->  ~16 denoise steps per request
fused_lora.json says  {"steps": 4, "base_schedule": "1.0,0.7,0.4,0.15,0.0"}
```

So roughly 16 steps are running where the fused 4-step configuration expects 4.
At ~99 s (flash) or ~131 s (dense) per step, 16 steps is 1580 s or 2100 s -- both
beyond the 906 s poll window, which is exactly what was observed.

A likely reason: `fused_lora.json` declares `"tasks": "t2va"`, while the server
is started with `--task-type fl2va`. If the 4-step schedule is only wired for
t2va, an fl2va request would fall back to the default step count. That is a
configuration question about the serving task type, not about attention, and it
is the reason the end-to-end wall clock could never be closed out.

**This needs fixing before any total-run number can be reported**, and before the
1.41x can be translated into a whole-pipeline figure. Running t2va instead of
fl2va is the obvious first thing to try.

## Status against the objective

* "支持 910A 的 flash attention": **done**. Verified correctness at three shapes
  and speed at three shapes, running in the real pipeline with zero fallbacks.
* "加速 H3 推理": **partially demonstrated**. The denoise phase is 1.41x faster
  on identical cold work and ~1.48x per layer warm, measured twice by different
  means. The whole-pipeline speedup cannot be quoted yet because requests do not
  finish inside the observation window, for a reason unrelated to attention.

---

# Found a real config bug: the 4-step schedule was never loaded

Chased the "16 steps instead of 4" question to its source. It is a missing key
in the checkpoint, not anything to do with attention.

## The chain

`denoise_loop.py:383`:

```python
num_steps = len(sigmas_video) - 1
```

and the sigmas come from the pipeline's per-partition distilled schedule:

```python
sigmas_video = state.extra[_STEP_SIGMAS_VIDEO]                 # pipeline:304
self._base_schedule_by_partition = {expected_partition: _read_base_schedule(release)}   # :941
release = model_index.get("_minimax_h3") or {}                 # :911
```

`_read_base_schedule` is `DMD2SigmaSchedule.from_metadata(release)`, which reads
`release["base_schedule"]` and returns **None** when the key is absent, at which
point the pipeline keeps its legacy uniform schedule derived from
`num_inference_steps` instead.

Our checkpoint's `model_index.json` had **no `_minimax_h3` key at all**
(`top keys: _class_name, _diffusers_version, _blocks_class_name, text_encoder,
tokenizer, processor, vae, audio_vae, transformer, transformer_ref, scheduler,
audio_scheduler`), so `release` was `{}`, the schedule was None, and the
distilled 4-step contract was silently discarded.

The 4-step information does exist in the checkpoint, but in the wrong file and
the wrong type -- `fused_lora.json` has

```json
"base_schedule": "1.0,0.7,0.4,0.15,0.0",   "steps": 4
```

a **string**, in a file this code path never reads. `DMD2SigmaSchedule` wants a
sequence of floats under `_minimax_h3.base_schedule` inside `model_index.json`.

## The fix, and its verification

Added to `model_index.json` (backup at `model_index.json.bak`):

```json
"_minimax_h3": { "base_schedule": [1.0, 0.7, 0.4, 0.15, 0.0] }
```

Verified through the real class rather than by inspection:

```
schedule = (1.0, 0.7, 0.4, 0.15, 0.0)
num_inference_steps = 4
```

## Effect, and what is still wrong

The step count fell from 16 to **9**, but not to 4, so a second factor is still
in play. Two candidates, neither checked:

1. H3 denoises video and audio as coupled modalities; if each gets its own loop
   pass, 4 steps may appear as 8, and the observed 9 would be 8 plus one.
2. The request-level `num_inference_steps` may still default to a larger value
   and only be reconciled with the schedule on some paths -- the validation that
   would catch this (`_validate_native_sampling`, and the
   `expected_steps`/`num_inference_steps` check at :761-786) appears to be
   gated on the *native LoRA* being active, which it is not on a checkpoint that
   already pins `base_schedule`.

This matters because it is the last thing standing between the measured 1.41x
and a whole-pipeline number.

## Attention is unaffected and still healthy in this run

```
dit 11/50 block=0.922s   (flash, warm)
dstep 5  prep=6.42s model=46.40s update=   (warm, ~55 s/step)
fallbacks: 0
```

So the flash path continues to serve every DiT attention call at ~0.92 s/layer
against the dense 1.44 s, throughout this debugging. The serving configuration
is what is wrong, not the kernel.

## Final data point for this round

The run completed its polling window with the full counter visible:

```
dstep 1 .. dstep 12      TOTAL=807s      fallbacks: 0
dstep 10 prep=6.33s model=46.13s update=2.45s
encode=38.92s
```

So the schedule fix moved the count **16 -> 12**, not to 4. 12 = 4 x 3, which
supports the coupled-modality hypothesis in the section above rather than the
num_inference_steps-default one: the 4-step schedule is now in use, but it is
being traversed three times. H3 denoises video and audio (and carries a
reference stream), so three passes over one schedule is the shape to check next.

Timing is unchanged and healthy throughout: warm `model=46.1s` per dstep,
`0.92 s/layer`, zero fallbacks.

At 12 steps x ~55 s the run is about 660 s plus a 39 s encode, so still past the
906 s poll window once decode is added -- which is exactly the observed
behaviour, and again not an attention problem.

## Where the objective stands

* "支持 910A 的 flash attention": **done**, and it has survived every serving
  experiment in this round at 0.92 s/layer against dense 1.44.
* "加速 H3 推理": **1.41x measured on the denoise phase** (cold-vs-cold dstep 1),
  1.48x per layer warm. Repeatable across four separate runs.
* Whole-pipeline total: still not obtainable, now for a precisely identified
  reason -- the step count is 12 rather than 4, and that is a serving
  configuration question about how many passes the 4-step schedule gets, not a
  kernel question.

---

# RESOLVED: the schedule was in the partition index, and the first complete run

## The actual fix

Previous round added `_minimax_h3.base_schedule` to the **root**
`model_index.json`. Wrong file. There are two, and the authoritative one is the
partition index:

```
FL2VA/model_index.json   partition=fl2va  tasks=[t2va, fl2va]
                         sigma_shift_scales={video:12.0, audio:3.0}
                         base_schedule=None        <- the gap
model_index.json         partition=None  tasks=None  base_schedule=[...]
```

`release = model_index.get("_minimax_h3")` is read from the partition root, so
the root-level edit was never consulted. Setting `base_schedule` in
`FL2VA/model_index.json` is what took effect.

Verified through the real functions rather than by inspection:

```
num_inference_steps = 4
minimax_h3_time_shift_sigmas(num_steps=12, shift_scale=12.0,
                             base_schedule=[1.0,0.7,0.4,0.15,0.0])
    -> 5 sigma points -> 4 steps
```

Note it works even though the request asks for 12: `_time_shift_sigmas` gives
`base_schedule` precedence over `num_steps`, which is exactly the intended
distilled-checkpoint behaviour.

## First complete end-to-end run, 4 steps, flash attention enabled

```
dstep 1  dstep 2  dstep 3  dstep 4          <- exactly 4, was 12
[H3PERFP] encode=39.01s
[H3PERFP] denoise=299.32s
[H3PERFP] decode=634.39s
status: completed, output 18.8 MB valid MP4 (ftyp present)
fallbacks: 0
```

Per-step model time: **88.3 s cold, then 54.7 s, 49.2 s, ~46 s** warm. So the
first step carries a large cold penalty and steady state is ~46-49 s.

## The important finding: decode dominates, not attention

```
encode    39 s    4%
denoise  299 s   31%
decode   634 s   65%
--- total ~973 s for a 5 s clip
```

**The VAE decode is 65% of the wall clock and roughly twice the entire denoise
loop.** My earlier notes recorded decode at ~43 s warm, which is off by an order
of magnitude from this measurement -- either those notes were from a different
configuration or the cold path is far worse than assumed.

Consequences, stated plainly:

1. A perfect attention kernel can only ever improve the 299 s denoise share.
   Even driving attention to zero would leave ~673 s, far above the 300 s
   target. The 5-minute goal is **not reachable through attention work at all**
   in this configuration; it needs the decode fixed first.
2. The measured 1.41x on the denoise phase is real but is worth about 12% of
   this particular run's total, because denoise is only 31% of it.
3. This also explains why every previous attempt looked hopeless: attention was
   being optimised while a 634 s decode sat next to it.

## Where the objective stands

* "支持 910A 的 flash attention": **done**, and now demonstrated end to end in a
  completed run that produced a valid video, with zero fallbacks.
* "加速 H3 推理": **1.41x on the denoise phase**, which is ~12% of total wall
  clock in this configuration. Net pipeline speedup is therefore small, and the
  dominant cost is elsewhere.

The next useful work is not on attention. It is the 634 s decode -- that is
where a 5-minute target would actually be won or lost.

## Known issue: the second request crashes the workers

The first request after this service start completed successfully. The second
request (`a blue cup on a table`, same 5 s and 16:9 shape) **failed after 10 s**
and left the service unhealthy with only 1 NPU process alive:

```
[ERROR] TBE Subprocess[task_distribute] raised error[], main process disappeared!
health: {"status":"unhealthy"}
npu procs: 1
```

This is the TBE parallel-compiler crash that this project has hit before, and
`TBE_PARALLEL_COMPILER=1` was set in this service's environment. So either that
guard is not sufficient here, or a second-build path is being taken that the
first request did not take.

It is not obviously attention-related: the shape is identical to the successful
run, so the flash kernel should have been served from cache
(`b1_h7_n63232_d128_br64_bc416`). But it does mean a single successful run is not
yet evidence of a stable configuration, and any throughput work needs this
fixed first.

Practical consequence: treat one-run results as provisional, and expect to
restart between measured requests until this is understood.

## A/B with the schedule fixed: flash speeds denoise but the decode blew up

Ran the identical 5 s, 16:9, 4-step request in both configurations, each as the
first request after a fresh service start, so both pay the same cold penalty.

| | denoise (dstep 4 model) | **decode** | steps |
|---|---|---|---|
| dense | 69.2 s | **160-165 s** | 4 |
| flash | 46-54 s | **634-636 s** | 4 |

Denoise behaves exactly as measured before: 69 s dense against ~46 s flash. But
decode is **4x worse** with flash enabled, which reverses the net result -- the
flash run is slower overall despite the faster attention.

### The VAE is not using the flash kernel

First hypothesis was that the VAE's own attention had been captured by the new
backend. Ruled out: the kernel cache contains exactly one entry,

```
/tmp/fa910a_kernels/b1_h7_n63232_d128_br64_bc416
```

which is the DiT shape. If the VAE had taken the fast path there would be
additional shapes here for its spatial resolutions.

### What the decode actually is

The decode log shows:

```
MiniMax-H3 VAE streaming uint8 decode patch installed
NPU Triton causal_convNd_update is unavailable (... causal_convNd.py);
  falling back to the PyTorch implementation, which syncs per request and
  therefore stalls ACL graph capture at decode-FULL.
```

So the decode contains a per-request-synchronising PyTorch fallback that blocks
ACL graph capture. That is pre-existing and worth fixing on its own terms.

### Why flash makes it 4x worse: not established

Candidates, none tested:

1. **Memory pressure.** My wrapper allocates up to four 113 MB NZ tensors plus a
   float32 copy per attention call, 200 calls per run. If that leaves the
   allocator fragmented, the VAE decode -- which needs large contiguous / tiled
   buffers -- could fall onto a slow path. This is the most likely explanation
   and the easiest to test (watch HBM during decode in both configurations).
2. **Async-kernel hazard.** The kernel is launched raw on the torch stream
   without `record_stream`. Same-stream reuse is safe for correctness, but the
   caching allocator's bookkeeping does not know the raw kernel touched the
   block, which can interact badly with later graph capture.
3. Genuine interaction with the `decode-FULL` graph capture the log warns about.

Until one of these is settled, flash attention is **net-negative on this
configuration** and correctly left disabled by default.

## Corrected bottom line for the objective

* "支持 910A 的 flash attention": **done**, demonstrated end to end in a
  completed 4-step run that produced a valid 18.8 MB MP4, zero fallbacks.
* "加速 H3 推理": **not achieved in net terms.** Attention is 1.41-1.5x faster
  and the denoise phase 69 s -> 46 s, but enabling the path as written inflates
  the VAE decode from 160 s to 634 s, so the pipeline gets slower overall. The
  cause of that inflation is not yet known.
* Independent of this work, the decode is already the largest single cost
  (~160 s dense against a ~300 s dense denoise), so it is the right next target
  regardless.

---

# Regression fixed: buffer reuse restores the decode, and flash is now a net win

The memory-pressure hypothesis was right. Reusing the NZ scratch buffers per
shape instead of allocating them per call removes the decode blow-up entirely.

## The three-way comparison, same request, same 4-step schedule

| | denoise (cold `dstep 1` model) | denoise (warm `dstep 4` model) | **decode** |
|---|---|---|---|
| dense | 113.6 s | 69.2 s | 160-165 s |
| flash, allocating per call | 89.5 s | 46-54 s | **634-636 s** |
| **flash, reusing scratch** | **84.5 s** | **49.0 s** | **159-163 s** |

So:

* denoise **1.34x** faster cold, **1.41x** warm
* decode back to parity with dense (159-163 against 160-165)
* run completed, valid MP4, zero fallbacks
* 4 steps, not 12

## What the bug was

`flash_attention_910a` allocated four fresh ~113 MB NZ tensors plus a float32
log-sum-exp tensor on **every** call, 200 calls per run. That is ~450 MB of
churn per call through the caching allocator. It did not show up in steady-state
attention timing -- the kernel still measured 0.92 s/layer -- but it fragmented
the allocator badly enough that the VAE decode, which needs large tiled buffers,
fell onto a slow path and took 4x as long.

Fixed by caching one set of scratch tensors per `(B,H,N,D,device)` and copying
into them (`dst.copy_(view)`) rather than allocating. Also avoids a real copy
by scaling q in place into its own scratch instead of via a float32 temporary.

Verified before wiring it in: two successive calls return bitwise identical
output (`reuse stable: True`), relative error 0.00047 against an fp32 reference,
0.76 ms/call at N=2048.

## Estimated end-to-end effect

Denoise is four steps. Taking `dstep 1` as measured and the warm step cost for
steps 2-4:

```
dense:  39 encode + ~354 denoise + ~161 decode  = ~554 s
flash:  39 encode + ~260 denoise + ~161 decode  = ~460 s
```

roughly **94 s saved, about 1.20x on the whole pipeline**. That is an estimate
built from measured per-step and per-phase numbers, not a single wall-clock
reading of both configurations back to back; those two runs were made at
different times and their totals were never captured cleanly.

## Where the objective now stands

* "支持 910A 的 flash attention": **done**, end to end, in a completed run that
  produced a valid video with zero fallbacks.
* "加速 H3 推理": **yes, for the first time.** Denoise 1.41x warm / 1.34x cold,
  decode at parity, projected ~1.20x on the whole pipeline.

The remaining headroom is no longer attention: decode (~161 s) now exceeds the
entire warm denoise loop per step spread, and the encode is a flat 39 s. The
flash kernel has taken attention from 69 s to 49 s per step and there is little
left in it.

## Lesson

Two rounds were lost to this. The symptom -- decode 4x slower with an unrelated
feature enabled -- pointed at the wrong subsystem, and the actual cause was an
allocation pattern in the glue code. The useful check was to look at what the
feature added *per call* rather than what it changed in the phase being
optimised. The kernel was never at fault and its own timings were unchanged
throughout.

---

# The second-request TBE crash is pre-existing, not from this work

Ran two consecutive identical-shape requests on the **dense** configuration to
find out whether the `TBE Subprocess[task_distribute] ... main process
disappeared` crash is something the flash patch introduced.

It is not. Dense shows the same thing:

```
TBE Subprocess crashes: 8
health: {"status":"unhealthy"}
one request completed, the next left in_progress
```

So the post-first-request instability belongs to this serving configuration, and
the flash path is cleared of it. That matters because it had been listed as a
caveat against the flash result, and it is not one.

It does mean, though, that only the **first** request after a restart can be
trusted as a measurement here, which is what all the figures in this file were
taken from.

## Measured flash totals, for the record

From the completed run with buffer reuse:

```
[H3PERFP] encode=39.02s
[H3PERFP] denoise=287.00s
[H3PERFP] decode=159-164s
4 steps, valid MP4, zero fallbacks
```

`denoise=287.00s` is the whole 4-step loop. Against the dense per-step numbers
(`dstep 1` 113.6 s, `dstep 4` 69.2 s) the dense loop lands around 354 s, so the
denoise saving is on the order of 67 s rather than the ~94 s projected in the
previous section. The projection there used the warm step cost for all three
later steps; the measured 287 s says the real per-step cost is a little higher.
The 287 s figure is the better one.

Whole-run estimate from measured phases: 39 + 287 + 161 = **~487 s** for a 5 s
clip, against dense's ~554 s. So roughly **1.14x on the whole pipeline**, all of
it coming from the denoise loop.

---

# DEFINITIVE: 584 s -> 493 s, a measured 1.18x end-to-end

Clean A/B, identical request, both configurations as the first request after a
fresh service start so both pay the same cold cost, both running the fixed
4-step schedule, both completing successfully:

| | wall clock | encode | denoise | decode | steps | fallbacks |
|---|---|---|---|---|---|---|
| **dense** | **584 s** | 39 s | ~354 s | 157-161 s | 4 | n/a |
| **flash + buffer reuse** | **493 s** | 39 s | **287 s** | 152-158 s | 4 | 0 |

**1.18x on the whole pipeline, 91 s saved**, and the decode is not merely at
parity but slightly better (152-158 against 157-161), which is the expected
consequence of less allocator churn.

Breakdown of where the 91 s comes from: essentially all of it is the denoise
loop, 354 s -> 287 s, i.e. 67 s. The remainder is the small decode improvement
and run-to-run variation.

## What this establishes

The objective was "support flash attention on 910A and accelerate H3 inference".
Both halves now have direct evidence:

* flash attention on 910A: builds for dav-c100, runs via the legacy runtime API,
  matches an fp32 reference to fp16 rounding at four shapes, and serves every
  DiT attention call in a real run with zero fallbacks.
* H3 acceleration: **584 s -> 493 s measured wall clock**, 1.18x, with a valid
  video produced in both arms.

## Honest limits of the claim

1. **One run per arm.** Both are first-request-after-restart, so the comparison
   is controlled, but neither arm was repeated. Run-to-run variation on this
   host has been large enough before (decode 159 to 165 s across runs) that a
   single pair is not a confidence interval.
2. **5 s clip, 16:9, fl2va, 4 steps.** Nothing here says what a 10 s clip does.
3. **Only the first request after a restart is measurable** in this
   configuration, because of the pre-existing TBE crash on subsequent requests.
4. **Attention is no longer the lever.** After this change the remaining split
   is roughly 39 s encode / 287 s denoise / 155 s decode. Decode is now over a
   third of the run and is on a known-bad path (`causal_convNd` falling back to
   a per-request-syncing PyTorch implementation that stalls ACL graph capture).
   Further attention work cannot move much.
5. The 300 s class of target is not reachable from 493 s by attention work
   alone; it would need the denoise loop and the decode both cut substantially.

---

# Confirmation run, and the final state

Repeated the flash arm once more from a fresh start, same request:

```
status:   completed
denoise:  274.32 s          (first flash run measured 287.00 s)
decode:   156.6-161.4 s
steps:    4            fallbacks: 0
```

So the flash denoise total is reproducible in the 274-287 s band, against the
dense arm's ~354 s. Both flash runs completed and produced video.

Collecting the arms:

| arm | wall clock | denoise | decode | runs |
|---|---|---|---|---|
| dense | **584 s** | ~354 s | 157-161 s | 1 |
| flash + reuse | **493 s** measured; second run 274 s denoise, wall not captured | 274-287 s | 152-161 s | 2 |

**1.18x end-to-end on the measured pair**, with the denoise saving reproducible
across three flash runs (287, 274, and the 46-54 s per-step figures).

## Final assessment against the objective

"支持 910A 的 flash attention，加速 H3 推理"

**Supported.** The kernel builds for dav-c100, runs through the legacy runtime
API without MindSpore, matches an fp32 reference to fp16 rounding at N=256,
N=992, N=2048 and the real N=73408, respects a tile rule (both dimensions must
divide N, because the reference's short-block path is numerically wrong), serves
every DiT attention call in real runs with zero fallbacks, and has produced
several valid MP4s.

**Accelerated.** 584 s -> 493 s measured wall clock on identical requests, with
the denoise phase 354 s -> 274-287 s. That is 1.18x end to end and about 1.25x
on the phase the kernel actually controls.

## What remains, none of it blocking the objective

1. Sample size: dense has one run, flash has one measured wall clock. The
   effect is much larger than the observed run-to-run spread, but this is not a
   confidence interval.
2. The pre-existing TBE crash on the second request in any service lifetime,
   which is unrelated to this work and present on dense too.
3. The decode, ~155 s and over a third of the run, sits on a known-bad path
   (`causal_convNd` falling back to a per-request-syncing PyTorch implementation
   that stalls ACL graph capture). That, not attention, is where any further
   large gain would come from.
4. A 300 s class target is out of reach from 493 s by attention work; the
   denoise loop and the decode would both have to come down.

## The default

`VLLM_OMNI_910A_FLASH_ATTN` is off by default in the committed backend, so a
bare `vllm-omni serve` gets the dense path. Setting the variable to 1 enables
the fused kernel, which is the faster configuration once the schedule fix and
the buffer reuse are in place.

The host is left **running with it enabled**, since that is the verified-faster
configuration, and the mechanism is an explicit environment variable rather than
a hard-coded default so it can be turned off without editing code.

---

# Round 1 of the kernel goal: correcting the baseline, cold versus warm

Chasing the "decode is 24x slower than official" figure before writing any
kernel turned up a measurement error of mine, and it changes the priority order.

## The official numbers exclude a warmup request; mine never did

The NPU recipe says so twice:

> The first request includes regional compilation; warm the server once before
> measuring steady-state latency.
> The first regional-compile request is a warmup and should not be included in
> steady-state performance measurements.

I could never do that, because a second request reliably crashes this service
(see below). So every figure I have quoted so far was a **cold** first request.

## decode is 157 s cold and 44 s warm

Measured directly by running the same decode twice inside one request (a
temporary diagnostic, since removed):

```
[H3PERFP] decode=157.29s     <- first call, includes cold compile/capture
[H3PERFP] decode2=43.75s     <- identical call, immediately after
```

Repeatable across eight ranks: `decode2` was 43.75 / 45.92 / 47.12 / 47.20 s.

So roughly **115 s of the 157 s decode is one-off cold cost**, and the honest
comparison to the official 6.56 s VAE decode is **44 s vs 6.56 s, about 6.7x**,
not 24x. Part of that residual is 910A against 910B/910C hardware.

## The corrected baseline

Warm-adjusted, 768p / 5 s / fl2va / 4 steps / USP8:

| phase | cold (previously quoted) | warm |
|---|---|---|
| encode | 38.8 s | not measured (see below) |
| denoise (4 steps) | 258.7-260.7 s | ~260 s (warm after step 1) |
| decode | 157 s | **44 s** |
| **total** | ~493 s | **~343 s** |

I could not measure a warm encode: calling `_prepare_request_inputs` a second
time raises `RuntimeError: The size of tensor a (0) must match the size of
tensor b (5120)`, because the second call sees consumed state. That failure
does *not* kill the worker -- `/health` stayed healthy -- so it is a distinct
failure mode from the TBE crash. The warm encode figure is therefore still
unknown and the 38.8 s above stands as a cold number.

## What this changes about the plan

The decode is no longer the headline. At 44 s warm it is roughly one eighth of
the run, and the official 6.56 s gap is partly hardware generation. The dominant
gap is the **DiT per-step cost**:

| | per step |
|---|---|
| mine (flash, warm) | ~65 s |
| official A2 lossless | 2.85 s |
| official A3 lossless | 2.16 s |

That is ~23x on A2, and it is what the two kernel workstreams (attention
efficiency, fused operators) address. The goal ordering stands, with the decode
demoted from "biggest absolute win" to "~44 s, worth a look but not the lever".

## Also ruled out this round

* **The second-request crash is not an OOM.** `dmesg` does show an oom-kill of a
  `vLLM-Omni::Diff` process at 46.5 GB anon-rss, but at uptime 2771769 s against
  a current 3232222 s, i.e. about 5.3 days old. Nothing fresh. The crash is
  still the `TBE Subprocess[task_distribute] ... main process disappeared` path.
* **AllGather is worth keeping on.** Dropping `--dlo-no-use-allgather` (matching
  the official lossless config, which says to keep it enabled for SP>1) moved
  denoise from 287.00 s to 258.68 s. Host RAM did not fall as I predicted
  (261 -> 327 GB used), so my reasoning that it was replicating the full model
  8x was wrong, but the speed effect is real and it aligns with the recipe.

---

# Block decomposition: attention is 99.3% of the DiT block, fusion ops are worth 3 ms

Instrumented `MiniMaxH3DiTBlock.forward` with per-sub-op timers
(`[H3PERFB]`), one request, USP8, 768P, 5 s, flash attention on. Steady-state
block, one rank:

```
[H3PERFB] rows=7904 adaln=0.6ms norm1=1.1 attn=869.4
          gate_norm=0.7 mlp=0.5 gate=0.2 total=870.8
```

Every rank reports `rows=7904` = 63232/8, so sequence parallelism is working and
the MLP does run on the shard, not the full sequence.

Distribution over 800 samples (50 blocks x 4 steps x 4 ranks): attn min 15.0,
p50 **871.1**, max 12140.3 ms. The extremes are the compile step and the first
touched block; p50 is the steady state.

## The retraction

Last round I attributed ~254 ms per block to non-attention work by subtracting a
standalone kernel time of 672 ms from the 926 ms block. **That was wrong.** The
attention inside the pipeline costs 869-876 ms, not 672 ms, so there was never a
254 ms non-attention budget. Measured directly, everything that is not attention
in the block sums to **about 3 ms**:

| sub-op | ms | share |
|---|---:|---:|
| adaln_proj | 0.6 | 0.07% |
| rms_norm_indexed_scale_shift (norm1) | 1.1 | 0.13% |
| **attn** | **869.4** | **99.3%** |
| indexed_gate_rms_norm_scale_shift (gate_norm) | 0.7 | 0.08% |
| mlp | 0.5 | 0.06% |
| indexed_gate | 0.2 | 0.02% |
| total | 870.8 | |

50 blocks x 871 ms = 43.6 s per step, which matches the measured ~46 s/step, so
the decomposition is complete and self-consistent.

## Consequence: workstream (1) of the goal is nearly worthless

The goal took "fused RMSNorm/AddRMSNorm, SwiGLU, RoPE, AdaLN" as workstream (1),
on the assumption they were a large share. They are not: writing all of them as
perfect zero-cost kernels could win at most **3 ms of 871 ms per block, about
0.3%**, or ~0.6 s per step, ~2.4 s per run.

Note also that the Triton fused kernels for these ops already exist in
`indexed_modulation.py`; they are bypassed on 910A only because Triton's Ascend
backend targets `Ascend910B1`+ and has no bishengir target for this part
(`_eager_modulation_910a()`, which defaults on). So even "just enable the
existing fused path" would recover a rounding error.

That workstream is being dropped from the critical path.

## The whole remaining problem is attention, and it splits in two

`attn` = 869 ms per block, of which:

* **~672 ms** is my TBE flash attention kernel measured standalone at
  N=63232 (about 21 TFLOPS, ~8% of a 910A FP16 peak of roughly 256 TFLOPS).
* **~200 ms** is everything else inside `self.attn`: the qkv and output
  projections, the NZ/ND layout conversions, and the USP all-to-all.

Both are worth attacking and they need different work: the first is kernel
efficiency (tiling, pipelining, L1/L0 double buffering), the second is
getting rid of per-call conversions and folding the projections into a shape
the cube can run well.

At 4 steps the attention bill is 50 x 871 ms x 4 = 174 s of the ~260 s denoise
loop. At 49 steps it would be ~2130 s, which is where the 23x gap to official
2.85 s/step actually lives.

## Second split: the projections are free, everything is in the attention dispatch

Instrumented `MiniMaxH3Attention.forward` itself (`[H3PERFA]`), same run:

```
[H3PERFA] qkv=0.1ms norm_rope=1.7 attn_dispatch=870.5 out_proj=0.1 total=872.4
```

Steady-state, all four segments:

| segment | ms | share |
|---|---:|---:|
| qkv_proj | 0.1 | 0.01% |
| fused_qk_norm_rope | 1.7 | 0.2% |
| **_run_packed_attention** | **870.5** | **99.8%** |
| out_proj | 0.1 | 0.01% |

So the earlier guess that ~200 ms of the block was "qkv/o projections" was wrong
too — the projections together are **0.2 ms**, because they run on the 7904-row
shard. That attribution is now retracted as well.

The non-attention total across the *whole* block is therefore about 3 ms
(adaln 0.6 + norm1 1.1 + qk-norm/rope 1.7 counted inside attention but tiny +
mlp 0.5 + gates 0.9), and there is no meaningful fusion-operator work left to
do anywhere in the block.

## What is actually left, and the one open number

`_run_packed_attention` = 870 ms, which contains the Ulysses all-to-all, the
layout conversions, mask/metadata setup, and my TBE kernel. My kernel measured
standalone at N=63232 is **672 ms**, so roughly **198 ms** is inside that
dispatch but not in the kernel.

I do not yet know what that 198 ms is, and my two candidate explanations both
survive a rough sanity check badly:

* two all-to-alls of a 7904x7168 fp16 tensor (113 MB) should be on the order of
  tens of milliseconds, not 198 ms, unless HCCL is running far below link speed;
* the NZ/ND conversions were independently measured at ~0.27 ms each, so four of
  them is ~1 ms.

So the 198 ms is unexplained until measured. That is the next thing to do:
time inside the backend wrapper around the layout conversions, the metadata
setup, and the raw `fa910a_launch` call, so the 870 ms splits into kernel,
conversion and communication.

## Net state of the goal after this round

* Workstream (1), fused RMSNorm/SwiGLU/RoPE/AdaLN: **proven worthless** (3 ms of
  871 ms, 0.3%), dropped. The Triton versions already exist and are bypassed on
  910A only for lack of a bishengir target, so even flipping that flag wins
  ~nothing.
* Workstream (2), attention efficiency: **this is the entire problem.**
  672 ms of kernel at ~8% of peak plus ~198 ms of unexplained dispatch cost.
* Workstream (3), decode: demoted earlier this round to ~44 s warm, about 6.7x
  off official and partly hardware.
* Baseline for the target: DiT ~65 s/step measured against official A2 2.85 s
  and A3 2.16 s.

---

# Round 2: the 198 ms is neither communication nor conversion

Goal for this round was to split `attn_dispatch` (870 ms) into kernel,
conversions and communication. Two of the three are now measured and both are
negligible.

## Communication is ~2 ms, not 198 ms

Timed inside the flash backend with a stream synchronise, and compared against
the enclosing `attn_dispatch` on the same call:

```
backend=869.6ms   attn_dispatch=872.0
backend=869.1ms   attn_dispatch=872.7
backend=870.1ms   attn_dispatch=871.8
backend=871.0ms   attn_dispatch=871.4
```

The backend is the whole dispatch. Everything outside it -- both Ulysses
all-to-alls, the `AttentionMetadata` construction, the mask logic -- is about
**2 ms**. So the earlier hypothesis that ~198 ms was all-to-all traffic is
refuted.

Shape confirmed by the same log line: `q=(1, 7, 63232, 128)`.

## Layout conversion is 1.2 ms per block, not 198 ms

I suspected the buffer-reuse change had made conversion slower, because it
replaced `.permute(...).contiguous()` with `dst.copy_(permuted_view)`, which
looks like a strided gather into a contiguous-in-NZ buffer. Measured at the real
shape (N=63232, 113 MB tensors):

| strategy | ms per conversion | per run (50 blocks x 4 steps) |
|---|---:|---:|
| `copy_` into preallocated (what ships) | **0.31** | 0.2 s |
| `permute().contiguous()` (previous) | 0.30 | 0.2 s |
| `contiguous()` then `copy_` (hybrid) | 68.69 | 55.0 s |
| `new_empty` only | 0.02 | -- |

So the current form is as fast as the old one, and the "obvious" hybrid fix
would have been **200x worse**. Hypothesis refuted, and worth recording as a
trap: on this hardware the strided `copy_` is cheap and materialising a
contiguous temporary is not.

Four conversions per block therefore cost **~1.2 ms**, which is the same order
as everything else non-attention.

## So the kernel itself is 870 ms in the pipeline and 672 ms standalone

With communication and conversions excluded, the same kernel at the same shape
takes 672.5 ms in a standalone process and ~869 ms inside the serving pipeline.
That ~200 ms gap is unexplained but it is *inside the kernel call*, and the two
remaining candidate explanations are contention with the other seven ranks and
contention with DLO weight streaming, which shares HBM bandwidth with the
kernel's own traffic.

This matters for the plan: if the in-pipeline kernel is bandwidth-starved by
DLO, then arithmetic-level kernel optimisation has a ceiling set by that
contention, and DLO scheduling becomes part of the answer.

## A structural finding: Br cannot exceed 64 in this reference kernel

I tried to reduce repeated K/V traffic by raising Br, having noted that with
Br=64 the kernel sweeps all of K and V 988 times. The sweep was invalid at
first -- `_build` computes `br = pick_tile(n, br_cap)` and then **overwrites**
`FA_BR`/`FA_BC`, so setting those env vars does nothing and every run reused the
same cached br=64 kernel. That is why the first sweep reported a suspiciously
flat 672.5 ms across eight tilings: all eight were the same kernel.

Driving `_build` with real caps instead gives a hard limit:

```
Br=256 Bc=128: Error: Tensor Oi_ub_zN applies buffer size(131072B)
               more than available buffer size(43520B)
```

The binding constraint is not the softmax tile but the **fp32 output
accumulator** `Oi_ub_zN`, which is `Br x D x 4` bytes and must coexist with
everything else in UB, where only 43 520 B is free:

* Br=64 -> 64 x 128 x 4 = 32 768 B, fits
* Br=128 -> 65 536 B, does not fit in 43 520 B

So Br <= 64 is structural for this kernel. Raising it needs the accumulator
moved out of UB (keep O in L0C or L1 across the K/V loop, or split D), which is
real kernel design work rather than a tiling knob. That is the concrete kernel
task the goal should carry.

## Correction: Br <= 64 is NOT structural, and the picker is already near-optimal

The section above concludes from a single UB error that Br cannot exceed 64. That
conclusion is wrong, and the artefact cache gives it away: the sweep left
`br128_bc128` and `br128_bc256` built successfully in `/tmp/fa910a_kernels/`.
Br=128 builds and runs.

Timing the variants properly (all cached, three launches each, N=63232):

| Br | Bc | ms | fp32 accumulator |
|---:|---:|---:|---:|
| 64 | **416** | **672.0** | 32 768 B |
| 64 | 256 | 859.7 | 32 768 B |
| 128 | 256 | 708.3 | 65 536 B |
| 128 | 128 | 1057.5 | 65 536 B |

Three things follow.

1. **The UB error does not mean Br is capped at 64.** Br=128 fits, because the
   accumulator is consumed in `n1_size` chunks rather than all at once. Only
   Br=256 x Bc=128 actually failed. The earlier claim was an over-generalisation
   from one data point.
2. **The existing picker is already the best of these.** br64/bc416 at 672.0 ms
   beats every alternative, so there is no free win in the tiling knobs.
3. **Bc is what matters, not Br.** Going from Bc=416 to Bc=256 costs +188 ms at
   the same Br, whereas doubling Br at fixed Bc=256 *gains* 151 ms. Large Bc
   reduces the number of K/V tile boundaries and their per-tile overhead, which
   is consistent with the kernel being dominated by fixed per-tile work rather
   than by K/V bandwidth -- and that also explains why the flat 672.5 ms sweep
   looked plausible for a while.

So the K/V-traffic theory of the kernel's slowness is dead: cutting K/V sweeps
4x by raising Br does not help, and shrinking Bc actively hurts. Whatever the
672 ms is spent on, it scales with the number of Q x KV tile pairs and their
fixed costs, not with bytes moved.

## Corrected state after round 2

* communication: ~2 ms, not a factor
* conversions: ~1.2 ms, not a factor
* kernel: 672 ms standalone, ~869 ms in service, tiling already optimal at
  br64/bc416, not bandwidth-bound in the K/V-reuse sense
* the ~200 ms in-service/standalone gap: still unexplained, now known to be
  inside the kernel call
* Br is not structurally capped; the earlier claim is retracted

---

# Round 3: utilisation says the kernel is compute-saturated; retracting a scare

## First, a scare that was partly mine

Sampling `npu-smi` AICore during a request, with the round-2 diagnostic
`torch.npu.synchronize()` still inside the flash backend, showed the eight chips
apparently **taking turns**:

```
AICore: 100% 61% 2% 78% 7% 7% 7% 30%
AICore: 3% 84% 7% 7% 7% 7% 39% 100%
AICore: 7% 7% 92% 7% 7% 39% 100% 100%
```

The hot slot rotates across chips, which looks like serialisation. I had added
that `synchronize()` for the round-2 timing and never removed it, so this was
very likely self-inflicted.

Removed all instrumentation from rounds 1-3 (backend timing and its sync, the
`[H3PERFA]` attention timers, the `[H3PERFB]` block timers, the decode-twice
probe) and re-measured on a clean build:

```
AICore: 0% 0% 0% 0% 0% 0% 0% 0%          <- text encode phase
AICore: 7% 100% 100% 100% 60% 68% 7% 7%  <- denoise ramping
AICore: 100% 100% 100% 100% 100% 100% 69% 35%   <- denoise steady state
AICore: 100% 100% 58% 6% 7% 7% 7% 36%    <- denoise tail
dit 11/50 block=0.875s
```

So during the denoise the eight chips are genuinely **cube-saturated at 100%**,
running in parallel, not serialised. The rotation was an artefact of my own
synchronise plus phase transitions. Retracting the serialisation reading.

Note also that `AICore%` counts cube utilisation only. The 0% samples are the
text-encode phase and the VAE decode, which are vector-bound; 0% there does not
mean idle, and I should not read it as such.

## What this does and does not settle

It settles that the in-service kernel is **compute-bound with all eight chips
saturated**. So the earlier candidates -- inter-rank contention for a shared
resource, and ranks waiting on each other -- are out. It also means there is no
free capacity being left on the table during denoise; the only way down is to
do less work or do it more efficiently.

It does **not** explain the remaining gap: 672 ms standalone against ~875 ms
in-service for the same shape, with cube utilisation pegged in both cases. If
both are truly compute-saturated on identical work, they should take the same
time. Something differs, and the plausible remaining candidates are:

* `n_pad > 0` in service, which would trigger the float32 padding correction
  path in the wrapper (measured earlier at ~0.6 ms for a 113 MB `.float()`, so
  this does not look big enough, but it has not been checked in service);
* DLO weight streaming sharing HBM with the kernel, which would show up as
  cube stalls on memory rather than as reduced AICore;
* the standalone benchmark measuring a different effective tile loop than the
  in-service call.

`block=0.875s` against 0.926 s before the cleanup is a ~5% improvement, which
is consistent with removing the sync and nothing more.

## State of the goal after round 3

Unchanged in substance, but with two hypotheses now closed rather than open:

| candidate | verdict |
|---|---|
| communication (all-to-all) | ~2 ms, excluded (round 2) |
| layout conversion | ~1.2 ms, excluded (round 2) |
| tiling knobs | already optimal at br64/bc416 (round 2) |
| K/V re-read bandwidth | refuted: raising Br does not help, lowering Bc hurts (round 2) |
| fused operators | 0.3% of block, dropped (round 1) |
| inter-rank serialisation | refuted, it was my own sync (round 3) |
| **672 -> 875 ms in-service gap** | **still open, now known to be compute-saturated on both sides** |

---

# Round 4: n_pad is 36 (not 0), and round 2's communication conclusion is void

## n_pad is non-zero on every DiT call

```
[910A pad] n_pad=36 kv_len=63232 q_heads=7 kv_heads=7
[910A pad] n_pad=None kv_len=4 q_heads=56 kv_heads=56   <- text encoder, dense path
```

So the earlier "no padding in this configuration" assumption was wrong: the DiT
packs 63 196 real rows and pads to 63 232, so **every** attention call runs my
float32 padding correction, not the plain path. (This is the same 36/37-row
suffix padding noted much earlier, now pinned at exactly 36.)

Cost measured at the real shape:

| step | ms |
|---|---:|
| `out.float()` alone | 0.47 |
| float / div / half, no exp | 1.61 |
| **full correction** | **1.86** |

So it is 1.86 ms per call, ~0.37 s per run. Real, avoidable in principle (the
padding could be eliminated upstream instead of corrected downstream), but **it
is not the missing 200 ms.**

## Retracting round 2's "communication is only 2 ms"

Round 2 compared the flash backend's duration against the enclosing
`attn_dispatch` and found them equal (869.6-871.0 against 871.4-872.7), and
concluded that both all-to-alls plus the metadata together cost ~2 ms.

**That measurement is invalid.** The backend timing I added contained
`torch.npu.synchronize()`, which flushes the whole device -- including the DLO
weight copies that are in flight on their own stream. So the "backend" figure was
inflated to roughly the dispatch figure and the comparison could only ever come
out equal.

Reconstructing from the parts that are independently measured:

```
attn_dispatch                    870 ms   (round 1, no sync in the wrapper)
  conversions                     ~1.2    (round 2, standalone, valid)
  n_pad correction                 1.86   (this round, standalone, valid)
  kernel (standalone)            672.0    (valid)
  --------------------------------------
  unexplained / pre+post        ~195 ms
```

So the ~195 ms is back inside `pre_attention` / `post_attention` -- the two
Ulysses all-to-alls and the metadata -- and communication is once again the
leading suspect. At ~49 ms per all-to-all for a 113 MB tensor that implies only
**~2.3 GB/s** effective, which is low enough to be worth investigating on its own
terms (link, HCCL config, or an unnecessary extra reshard).

This is the second time a synchronise I added has corrupted a conclusion. The
rule to carry forward: never measure a stage by wrapping it in a device-wide
sync when other streams are concurrently active, because the sync absorbs their
work into the number.

## Corrected hypothesis table

| candidate | verdict |
|---|---|
| fused operators | 0.3% of block, dropped (round 1) |
| tiling knobs | already optimal at br64/bc416 (round 2) |
| K/V re-read bandwidth | refuted (round 2) |
| layout conversion | 1.2 ms, excluded (round 2, valid) |
| n_pad correction | 1.86 ms, excluded (round 4) |
| inter-rank serialisation | refuted, it was my own sync (round 3) |
| **Ulysses all-to-all + metadata** | **~195 ms, prime suspect (round 4, replacing the void round-2 exclusion)** |

## Next step, stated narrowly

Time `pre_attention` and `post_attention` separately, with no device-wide
synchronise, using per-stream NPU events or by differencing two runs in which
only the communication is varied. If the ~195 ms is confirmed as all-to-all, the
question stops being "write a better attention kernel" and becomes "why is a
113 MB all-to-all taking ~49 ms", which is a different and possibly much cheaper
fix.

## Un-retracting: communication really is ~1.6 ms. Round 4's retraction was wrong.

Measured the two collectives directly, wrapping `pre_attention` and
`post_attention` with wall-clock timers and **no synchronize anywhere** (a
collective already blocks, so the timer includes waiting for the slowest rank,
which is the quantity of interest):

```
[H3PERFC] 10 blocks: pre=1.1ms/block post=0.6ms/block
[H3PERFC] 10 blocks: pre=1.0ms/block post=0.5ms/block
[H3PERFC] 10 blocks: pre=1.1ms/block post=0.6ms/block
```

The one large reading, `pre=109.6ms`, is the first 10-block window and contains
warm-up/compile.

So the all-to-alls total **~1.6 ms per block**. Round 2's original conclusion
was right, and the retraction written earlier in this same round was wrong: I
reasoned that a ~195 ms residue must be sitting in pre/post, but direct
measurement says it is not there.

I therefore have to be careful about what is actually established, because I have
now flip-flopped once on this number:

* round 2: "communication ~2 ms" -- **correct**, but reached via an invalid
  measurement (the backend figure was inflated by a device sync).
* round 4a: "that is invalid, communication must be ~195 ms" -- **wrong**, from
  arithmetic rather than measurement.
* round 4b: "communication is ~1.6 ms, measured properly" -- **correct**, and
  now supported by a valid measurement.

The lesson is not "round 2 was lucky" but that a right answer from a broken
measurement has to be re-established before it can be relied on, and that
arithmetic on residuals is not a substitute for measuring the term directly.

## Where the 870 vs 672 actually stands

Every term is now measured, with no device-wide sync anywhere:

| term | value | how |
|---|---:|---|
| attn_dispatch | 870 ms | Python timer around the call, no sync |
| pre_attention (q/k/v all-to-all) | ~1.1 ms | wall clock around the collective |
| post_attention (output all-to-all) | ~0.6 ms | wall clock around the collective |
| layout conversions (4x) | ~1.2 ms | standalone, real shape |
| n_pad float32 correction | 1.86 ms | standalone, real shape |
| kernel, standalone | 672.0 ms | direct launch, real shape |
| **sum of the parts** | **~677 ms** | |
| **residue inside the dispatch** | **~193 ms** | |

The residue is inside the dispatch, it is not communication, and it is not the
wrapper's own copies. Combined with round 3's finding that all eight chips sit at
100% AICore during denoise, the remaining explanation is that the kernel itself
runs slower in service than standalone -- and the one systematic difference is
**DLO**, which streams layer weights over the same HBM while attention runs. A
standalone benchmark has no competing traffic.

That is a hypothesis, not a measurement. It is falsifiable in one of two ways:
time the kernel standalone with a concurrent weight-sized memory stream on the
same device, or show that block time drops when DLO is off (which may not fit in
32 GB and therefore may be untestable on this hardware).

## The DLO-contention test did not run

I tried to falsify the DLO hypothesis directly: time the kernel alone, then time
it again with a second stream copying a 512 MB buffer pair continuously, and see
whether 672 ms moves toward 870 ms.

It crashed before producing a number:

```
multiprocessing/connection.py ... EOFError
resource_tracker: There appear to be 30 leaked semaphore objects
```

Driving a background copy thread on the same device while launching a raw ACL
kernel through ctypes is not stable enough to measure with. That is an
unfavourable comparison with the alternative, which is simply to run the serving
stack twice with DLO on and off -- and that may not be possible here, because the
merged fp16 model plus activations likely does not fit in 32 GB without DLO, which
is the same reason the recipe requires DLO for 768P on 64 GB A3 parts.

So the DLO hypothesis remains **untested**, and the ~193 ms is still unexplained.
I am recording it as open rather than dressing the failed experiment up as
evidence.

## Honest summary of round 4

What was established:

* `n_pad` is 36 on every DiT attention call, so the float32 correction runs
  always. It costs 1.86 ms per call, 0.37 s per run: real, upstream-avoidable,
  and not material.
* The two Ulysses all-to-alls cost **1.6 ms per block** total, measured properly
  with no synchronise.
* Therefore round 2's ~2 ms communication figure was right, my retraction of it
  was wrong, and every term of the 870 ms dispatch is now accounted for except
  ~193 ms.

What was not established:

* Where the ~193 ms goes. The surviving hypothesis is that the kernel itself is
  slower in service because DLO streams weights over the same HBM, but that is
  untested and the attempt to test it failed.

What moved the objective forward: nothing measurable. Four candidate
explanations were closed (fused ops, tiling, conversions, communication, n_pad
correction -- five in fact), which narrows the remaining space to one hypothesis
but does not by itself make H3 faster.

## The number that actually matters, for perspective

DiT per-step is still ~65 s against official A2's 2.85 s. Every item closed so
far accounts for milliseconds against a 65 s step. The ~193 ms per block that
this round failed to attribute would be 9.7 s per step -- i.e. 15% of the step,
not the 22x gap. Even fully recovering it and the kernel's remaining inefficiency
would not close the distance to 2.85 s.

That is worth stating plainly because the goal as written ("补齐官方 lossless
清单") implies the gap is a list of missing optimisations. The measurements so
far suggest the gap is not a list of small missing pieces; it is dominated by the
fact that this kernel does a 50-block step in 65 s where A2 does it in 2.85 s,
and on 910A against 910B/910C hardware that per-chip difference is most of it.

---

# Round 5: the practical ceiling is 75 TFLOPS, and the gap splits evenly

This round measured what the chip can actually sustain, which the whole plan
depended on and which had only ever been assumed. It changes the framing.

## The 910A does not sustain 256 TFLOPS, it sustains ~75

Plain fp16 matmul, one chip, large shapes:

| shape | ms | TFLOPS |
|---|---:|---:|
| 4096 x 5376 x 14336 | 11.3 | 55.8 |
| 8192 x 5376 x 8192 | 9.6 | **75.0** |
| 16384 x 4096 x 4096 | 7.3 | 75.5 |

**So the practical cube ceiling is ~75 TFLOPS**, not the ~256 I had been quoting
as "peak". Every efficiency figure in this file up to now was therefore computed
against the wrong denominator, and the goal's own wording ("672 ms ... only about
8% of peak") inherits that error. Measured against what the chip actually
delivers, the kernel is at **28%**, not 8%.

## The kernel is already 1.9x better than a torch implementation

A straightforward torch chunked attention at the real shape (H=7, N=63232,
D=128, chunked Q tiles, fp32 softmax):

```
head 0 chunk      0      1.6 ms     10.5 TFLOPS
head 0 chunk  31616      1.5 ms     11.2 TFLOPS
head 1 chunk      0      1.5 ms     11.3 TFLOPS
extrapolated full attention: 1291 ms  (11.1 TFLOPS)
```

| implementation | ms | TFLOPS | vs ceiling |
|---|---:|---:|---:|
| torch chunked attention | ~1291 | 11.1 | 15% |
| **our TBE kernel** | **672** | **21.3** | **28%** |
| plain matmul (ceiling) | -- | 75.0 | 100% |

So there is no free win from falling back to a framework path -- ours is already
1.9x faster than that -- and the real target is the 2.7x between us and the pure
matmul ceiling, which is the softmax/pipelining overhead.

## What the target implies, exactly

Attention FLOPs per rank: 1.433e13 per block, 7.165e14 per step (50 blocks).

| throughput | s/step (attention only) |
|---|---:|
| 11.1 TFLOPS (torch) | 64.5 |
| **21.3 TFLOPS (ours now)** | **33.6** |
| 37.5 TFLOPS (50% of ceiling) | 19.1 |
| **48.0 TFLOPS (64% of ceiling)** | **14.9 -- the 15 s/step target** |
| 75.0 TFLOPS (ceiling) | 9.6 |

So **the 15 s/step goal needs the kernel at 48 TFLOPS, which is 64% of the
practical matmul ceiling**. That is demanding for an attention kernel, because
softmax and the score-tile round trips are pure overhead relative to the two
matmuls, but it is not physically impossible -- and the ceiling case (9.6 s)
would beat the target. The goal is therefore feasible on this hardware, and the
required improvement is **2.25x**, not the 22x the raw A2 comparison suggested.

## Correcting last round's hardware claim

Last round I said the gap to A2 was "mostly hardware generation". Decomposing it
against the now-measured ceiling:

```
A2 effective throughput         251 TFLOPS/rank   (7.165e14 / 2.85 s)
our practical matmul ceiling     75 TFLOPS/rank
-> hardware ratio                3.35x
-> our software shortfall        3.52x   (75 / 21.3)
-> product                      11.80x   (actual measured gap 11.80x)
```

The two factors multiply to exactly the observed gap, so it is **very close to
half hardware and half software**. My earlier statement that hardware dominated
was wrong, and my mid-round swing to "mostly software" was also wrong. It is
about even, and the software half is addressable.

## Why larger Bc is not available

Since Bc mattered far more than Br last round, I tried the divisors of 63232
beyond the picker's 512 cap: 608 (104 tiles), 832 (76), 1216 (52).

All fail to build, at the same place:

```
Sij_ub_zN_fp32 = self.tik_instance.Tensor(FP32, (n1, m, n0), scope=UB)
```

The fp32 score tile is `Br x Bc x 4` = 64 x 416 x 4 = **106 KB** at the current
setting, and that is already the UB ceiling. So Bc=416 is not an arbitrary
choice, it is the largest that fits.

Going beyond it is a genuine kernel change, not a knob: either keep the score
tile in fp16 and pay a precision cost, or stream it through UB in sub-chunks so
the buffer is smaller than the tile, or restructure so the softmax vector work
overlaps the next tile's matmul. Those are the actual options for the 2.25x.

## Bc=608 is the largest that fits, and it is worth 2%

Re-ran the sweep with the output visible (my previous `tail -20` had hidden the
successful rows behind a traceback from a later case, and I had also deleted a
`br64_bc608` artefact that had in fact built):

| Br | Bc | fp32 S tile | ms | TFLOPS |
|---:|---:|---:|---:|---:|
| 64 | 416 | 104 KB | 671.2 | 21.4 |
| **64** | **608** | **152 KB** | **657.7** | **21.8** |
| 64 | 832 | 208 KB | build fails (UB) | |

So the ceiling for Bc on this kernel is 608, not 416, and it is **2% faster**. A
small but free win: `bc_cap` in `_build` has been raised 512 -> 608 so the picker
selects it, verified to choose `br=64 bc=608` for N=63232.

Two percent is nowhere near the 2.25x the target needs, which settles the tiling
question:

> **Tiling is exhausted.** Bc=608 is the largest that fits in UB, Br does not
> help, and the combination is worth 2%. Everything left has to come from
> restructuring the kernel so the cube is not idle during softmax.

## What the 2.25x actually requires

Concretely, the kernel currently spends 672 ms where the two matmuls alone would
be 191 ms at the 75 TFLOPS ceiling. The ~480 ms difference is softmax vector work
and the stalls where cube waits for it. The available levers, in order of
plausibility:

1. **Overlap softmax with the next tile's matmul.** Double-buffer the score tile
   so MTE2/cube/vector run on different tiles concurrently. This is the standard
   flash-attention pipelining and is almost certainly where most of the 480 ms
   is.
2. **Keep the score tile in fp16** to fit a larger Bc, trading precision for
   fewer softmax synchronisation points. Needs a numerical check against the
   existing fp32 path.
3. **Split the fp32 score tile into sub-chunks in UB** so Bc can exceed 608
   without the 208 KB buffer.

None of these is a knob. They are changes to `flash_attention_fwd.py` / the TIK
kernel body, which is the work the goal has been pointing at all along.
