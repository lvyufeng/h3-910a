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

---

# Round 6: found WHY the kernel is at 28% -- it has no software pipeline at all

## The cube is idle about 40% of the time

Ran the kernel in a loop on an otherwise idle chip and sampled AICore:

```
AICore: 100% 49% 7% 7% 7% 7% 100% 100% 100% 100%
RAN 155 launches in 101.8 s -> 657.1 ms/launch, 21.8 TFLOPS
```

So the cube duty cycle is roughly **55-60%**. At 100% duty the same work would
take ~375 ms rather than 657. The gap to the 191 ms matmul-ceiling figure is
therefore two separate problems: cube idle time, and cube efficiency per matmul
(small K=128 tiles).

## The cause: no pipelining and no synchronisation primitives anywhere

Grepped the TIK kernel body for the machinery a flash-attention software pipeline
needs:

```
double_buf | _dbl | buf_num | DOUBLE | ping | pong   ->  0 hits
set_scope  | wait_flag | set_flag | set_atomic      ->  0 hits
```

Neither double buffering nor any manual flag-based synchronisation exists. In TIK
that means the three hardware pipelines -- MTE2 for GM->L1 loads, the cube for the
two matmuls, and the vector unit for softmax -- can only issue in program order
with implicit full barriers between them:

```
load Vj, Kj   -> wait for load to finish
qk matmul     -> wait for cube to finish
softmax       -> wait for vector to finish
pv matmul     -> wait for cube to finish
```

Each stage drains before the next begins, so at any instant at most one of the
three pipelines is doing work. That is exactly the 55-60% duty cycle, and it is
the dominant term in the 480 ms of non-matmul time.

## This is the actionable root cause

The fix is the standard flash-attention software pipeline, and it is now a
specific, bounded piece of work rather than a search:

1. Double-buffer `Kj_l1_zN` / `Vj_l1_zN` so MTE2 prefetches tile N+1 while the
   cube works on tile N.
2. Double-buffer `Sij_ub_zN` (fp32) so the vector unit softmaxes tile N-1 while
   the cube computes tile N.
3. Insert `set_flag` / `wait_flag` pairs between the three pipelines with distinct
   event ids per buffer, so the compiler stops serialising them.
4. Keep the existing Br=64 / Bc=608 tiling, which is already the largest that
   fits in UB; the pipeline must therefore fit its extra buffers into the same
   budget, which is the main design constraint.

Expected gain: recovering the idle time alone takes 657 -> ~375 ms (1.75x). Better
cube efficiency on the small-K matmuls is a further, separate opportunity. The
2.25x needed for 15 s/step is therefore reachable in two steps rather than one
heroic one.

## What this round changed

It converted "the kernel is 2.7x off its own hardware ceiling, cause unknown"
into "the kernel has no software pipeline, here is the buffer to double, here are
the flags to add". That is the difference between research and engineering, and
it is the first round where the remaining work has a concrete shape.

Ital also settles a question from round 2: the flat 672 ms across all tilings was
not evidence about bandwidth at all. With no pipelining, time is dominated by
serialised stage latencies, which are nearly tiling-insensitive -- which is
exactly what a flat curve looks like.

## The ceiling was measured at the wrong K, and that revises feasibility again

The 75 TFLOPS ceiling was measured at K=5376. The attention kernel's matmuls are
nothing like that. Measured TFLOPS against reduction length:

| M | K | N | ms | TFLOPS |
|---:|---:|---:|---:|---:|
| 4096 | 5376 | 8192 | 4.94 | **73.0** |
| 4096 | 2048 | 8192 | 1.98 | 69.5 |
| 4096 | 608 | 4096 | 0.39 | 51.7 |
| 4096 | 512 | 8192 | 0.65 | 52.8 |
| 4096 | **128** | 8192 | 0.34 | **25.0** |
| 8192 | **128** | 8192 | 0.65 | 26.4 |
| 2048 | **128** | 63232 | 1.19 | 27.8 |

`QK^T` has K = D = 128 and that is **not changeable** -- it is the head dimension.
`PV` has K = Bc = 608. So the two halves of the attention have different ceilings:

```
QK^T (K=128) at ~26 TFLOPS : 13.8 s/step
PV   (K=608) at ~52 TFLOPS :  6.9 s/step
shape-limited total        : 20.7 s/step   (34.6 TFLOPS effective)
```

Consequences, stated plainly:

* We are at **62% of this shape-limited ceiling**, not 28% of a 75 TFLOPS one. The
  kernel is closer to what a vendor matmul achieves at these shapes than the
  earlier framing suggested -- a vendor-level matmul at K=128 only reaches
  25-28 TFLOPS, and we get 21.8 overall while also doing softmax.
* **The 15 s/step target needs 47.8 TFLOPS, which is 1.38x ABOVE the shape-limited
  ceiling.** So pipelining alone cannot reach it. Even a perfect pipeline that hit
  the vendor matmul's short-K efficiency would land near 21 s/step.
* Reaching 15 s would require beating the vendor library's K=128 efficiency by
  more than 2x, i.e. exactly the kind of gain MindIE-SD's LaserAttention appears
  to deliver (see below), from a hand-written fused kernel.

## And the A2 comparison implies a startling number

At 2.85 s/step with the same per-rank work, A2 must be sustaining **251
TFLOPS/rank**. That is **7.3x our shape-limited ceiling of 34.6 TFLOPS**. Even
allowing generous hardware generation differences, the bulk of that cannot be
silicon -- it has to be that LaserAttention extracts far more from a K=128
matmul than either our kernel or the vendor library does.

I do not know how, and I am not going to guess in this file. What it does
establish is where the remaining gap actually lives: not in pipelining our
existing structure, but in a fundamentally better short-K attention
implementation.

## Corrected position on the goal's 15 s/step target

Two rounds ago I said the target was reachable and needed 2.25x. With the ceiling
measured at the real K it is **not reachable by the path I proposed**. The honest
statement:

* pipeline recovery: 33.6 -> ~21 s/step (best case, ~1.6x) -- feasible, worth doing
* 15 s/step: needs 47.8 TFLOPS against a 34.6 TFLOPS shape-limited ceiling -- not
  reachable without beating the vendor short-K matmul by >2x
* official 2.85 s/step: needs 251 TFLOPS -- a different class of kernel

So the plan for the next rounds is to do the pipeline work because it is real and
worth ~1.6x, but the 15 s/step figure in the goal should be treated as
out of reach on this hardware unless the short-K efficiency can be broken through.
I would rather say that now than after another twenty rounds.

---

# Round 7: the 39 s text encode is fixed per-request offload cost, not compute

Switched to goal item (3) this round, because the kernel side has a measured
ceiling (round 6) and the encode had never been looked at despite being 39 s of a
~343 s run against an official 0.48 s.

## It is entirely text encoding

Split `_prepare_request_inputs`:

```
[H3PERFT] text_enc=39.10s cond_prep=0.00s
[H3PERFP] encode=39.23s
```

So the whole 39 s is `_prepare_local_conditioning`; the latent/conditioning
preparation is 0.00 s.

## It is fixed overhead, not prompt-dependent work

Submitted a 378-word prompt against the short-prompt baseline:

| prompt | text_enc |
|---|---:|
| ~4 words | 39.10-39.23 s |
| **378 words** | **39.17 s** |

Identical. A 50-layer Qwen3-VL forward over the actual tokens cannot be
prompt-length-insensitive like that, so the 39 s is **per-request fixed cost**,
not the encoder doing more or less work.

## The cause: the text encoder is staged-offloaded and re-gathered every request

The served log says:

```
Prepared text_encoder for pipeline-managed staged offload
Prepared video_vae  for pipeline-managed staged offload
Prepared audio_vae  for pipeline-managed staged offload
```

and the mechanism is `vllm_omni/diffusion/offloader/component_utils.py`:

```python
def prepare_component(module, name, *, device, stage_on_demand, blockwise, staged_components):
    if stage_on_demand:
        validate_on_demand_component(module, name)
        getattr(module, "offload_to_cpu")()          # <- evicted from device
        staged_components.append(module)
        logger.info("Prepared %s for pipeline-managed staged offload", name)
    elif not blockwise:
        module.to(device)
```

called as `stage_on_demand=component.on_demand` from `component_utils.py:172/182`,
where the component classification comes from `ResolvedOffloadPlan` in
`offloader/plan_resolver.py`. That plan has separate `dits`, `encoders`, `vaes`
and **`residents`** tuples, so "keep this resident" is an existing concept in the
design.

With DLO enabled the encoder is therefore evicted to host between requests and
has to be brought back and TP-gathered across all 8 ranks on the next one. That
is the fixed ~39 s, and it explains why the official A2/A3 numbers are 0.48-0.83 s
-- on 64 GB parts the encoder does not need to be evicted at all.

## Why this matters more than the kernel work

* 39 s on a ~343 s warm run is **11% of the pipeline**, and removing it is a
  configuration/classification change rather than a kernel rewrite.
* Unlike the attention work, this one has no measured ceiling in the way: the
  target of 2 s is plausible if the encoder simply stays resident, because the
  compute itself is evidently cheap (it does not scale with prompt length).
* HBM was only ~8.7 GB used of 32 GB during denoise, so there may well be room to
  hold the encoder resident; that has to be checked rather than assumed, since
  DLO's own pool is the thing being sized.

## Next step, narrowly

Find where `ResolvedOffloadPlan` classifies `text_encoder` as `on_demand`, and
put it in `residents` instead (or find the config that already expresses that).
Then measure `text_enc` again: the expectation is that it drops to roughly the
real forward cost, which the prompt-length experiment says is small. If HBM does
not permit it, the fallback is to keep the encoder's weights on device but its
activations transient, which is a smaller change than the current full eviction.

## Round accounting

No speedup landed this round -- the instrumentation is removed and the service is
stopped. What was established is a named, bounded cause for an 11% slice of the
pipeline, with the exact function and the exact knob. Given that the attention
path is against a measured ceiling, this is the more promising of the two
remaining fronts.

---

# Round 8: the offload hypothesis was wrong; the encoder is on NPU and still takes 39 s

Round 7 concluded the 39 s encode was staged-offload cost and proposed keeping the
text encoder resident. That was tested this round and **refuted**, twice, before
the real state of affairs became clear.

## Test 1: take text_encoder out of on_demand

The declaration at `pipeline_minimax_h3.py:639` is

```python
on_demand_component_paths=frozenset({"text_encoder", "video_vae", "audio_vae"}),
```

and `prepare_component` evicts anything that is `stage_on_demand`. So I removed
`text_encoder` from that set. The served log confirmed the change took effect:

```
Prepared audio_vae for pipeline-managed staged offload
Prepared video_vae for pipeline-managed staged offload      <- text_encoder gone
```

**Result: encode was 39.85 s, unchanged.** So the eviction was not the cause.

## Test 2: where do the weights actually live?

Flat timing regardless of prompt length had two candidate explanations -- a fixed
padded sequence length, or the encoder running somewhere slow. Logged the first
parameter's device at the point the encoder runs:

```
[H3PERFD] text_encoder params device=npu:0 dtype=torch.bfloat16
[H3PERFP] encode=39.54s
```

So the encoder is **already on the NPU, in bfloat16**, and still takes 39.5 s. That
kills both the "it is on CPU" and the "it is a host-to-device transfer" stories.

## What is actually true

The text encoder is registered for **blockwise layer streaming**, a different
mechanism from `on_demand`:

```python
encoder_block_attrs={"text_encoder": ("vision.blocks", "text_model.layers")},
```

so its layers are streamed through the device layer by layer during the forward,
and `on_demand` was never the relevant switch. That is why removing it from
`on_demand_component_paths` did nothing.

The 62.1 GB on-disk size divided by the 39 s gives ~1.6 GB/s, which is exactly a
plausible host-to-device streaming rate -- so the timing is consistent with the
encoder's layers being streamed in on every request rather than already resident.
But note that is now an inference again, not a measurement, because the direct
measurement (params on npu:0) shows at least the first parameter resident. I have
not established whether the whole encoder is resident during the forward or
whether it is being re-streamed per layer, and the two imply very different fixes.

## Honest status

* The round-7 conclusion ("39 s is staged-offload cost") is **withdrawn**.
* The measurement that would settle it -- whether HBM rises by ~7.8 GB while the
  encoder runs, i.e. whether all 8 ranks hold their shard simultaneously -- was not
  taken. HBM sat at 8.6-8.8 GB per chip during a full request, which is too low
  for a resident 62/8 = 7.8 GB encoder on top of the DiT pool, which argues for
  re-streaming; but "too low to be resident" is an inference from a number I
  sampled once, not a controlled measurement.
* No speedup landed. The on_demand change was reverted since it demonstrably did
  nothing, and all instrumentation was removed.

## Next step, narrowed again

Sample per-chip HBM at ~1 s resolution across a whole request and look for the
encoder's ~7.8 GB appearing and then disappearing. That single trace distinguishes
"resident, 39 s is compute on a padded sequence" from "re-streamed every request,
39 s is transfer", and those have completely different fixes -- the first needs
the padded length found and cut, the second needs the layer streaming disabled for
the encoder specifically.

That trace is the cheapest decisive thing left in this thread, and it should have
been the first thing done rather than the third.

## Also: my process check was broken

Twice this round `pgrep -cf "[v]llm..."` reported large counts (154, then 2) that
were really the pattern matching my own shell command line. The host was clean all
along: NPU processes 0, DiffusionWorker 0, RAM settled at 9 GB used. Worth
recording because a false "orphans still running" reading could easily have led to
a needless reboot or to conclusions about memory pressure.

---

# Round 9: the HBM trace settles it -- the encoder is streamed, not resident

The measurement that should have been first. Sampled per-chip HBM every ~1.5 s
across a whole request, timestamped from submission.

```
 t(s)   HBM chip0 / others (MB)
  0.5   3414 3222 ...           idle
 12.3   4228 4036 ...           encoder begins streaming in
 16.3   5324 4760 ...
 18.3   5324 5132 ...           plateau: encode running
 35.9   5330 5138 ...
 37.9   4608 4416 ...           encode ends, buffers released
 41.8   4752 4560 ...
 71.1   7076 6884 ...           denoise / DLO pool grows
 82.8   8866 8674 ...           denoise steady state
155.6   9102 8690 ...
```

## What it proves

* During encode HBM peaks at **5.33 GB**, i.e. an increment of only **~1.9 GB**
  over the 3.4 GB idle baseline. A resident TP8 shard of the 62.1 GB encoder would
  be **~7.8 GB**. It is not there.
* At t=37.9 s, exactly when encode finishes, HBM **drops** by ~0.7 GB. Resident
  weights would not be released at that moment; per-layer streaming buffers would.
* Denoise then settles at 8.87 / 8.68 GB and stays flat.

So the text encoder is being **streamed through the device layer by layer on every
request**, and the 39 s is the transfer. 62.1 GB / 39 s = 1.6 GB/s, consistent with
a host-to-device rate.

This confirms the mechanism suspected last round via `encoder_block_attrs` and
finally refutes both earlier stories (staged-offload in round 7, CPU/NPU placement
in round 8). The encoder was never resident, and `on_demand` was never the switch.

## And residency would fit

Denoise steady state is 8.87 GB of a 32 GB chip. A resident encoder shard is
~7.75 GB. Together that is **~16.6 GB**, leaving ~15 GB headroom. So the fix is
available on this hardware, contrary to my round-7 reflex of comparing 62 GB
against 32 GB and forgetting the TP8 sharding.

## The fix, concretely

Two things have to change together, because `prepare_component` skips
`module.to(device)` when `blockwise` is true:

1. drop `text_encoder` from `encoder_block_attrs` in the `_offload_plan`
   declaration (`pipeline_minimax_h3.py:636`), so its layers stop being streamed;
2. make sure it is actually placed on the device once, since the blockwise branch
   is what currently prevents the `elif not blockwise: module.to(device)` from
   running.

Expected: `text_enc` falls from ~39 s toward the real forward cost. The
prompt-length experiment (4 words and 378 words both 39 s) says the compute part is
small, so the target of ~2 s in the goal is plausible rather than optimistic.

## Round accounting

No speedup landed, again. But this round converted a two-round-wrong hypothesis
into a measured mechanism plus a specific two-line change with a favourable memory
budget. The trace took four minutes to run and would have saved rounds 7 and 8 had
I done it first; that is the real lesson of this stretch.

## The proposed fix does not work either: 39 s -> 42.85 s

Applied the two-part fix from the section above -- emptied `encoder_block_attrs` so
`text_encoder`'s `("vision.blocks", "text_model.layers")` entry stops driving
per-layer streaming -- and measured:

```
idle HBM 3194 / 3002 MB
 t=35s   4388 / 4196 MB      during encode
 t=45s   4512 / 4320 MB
encode=42.85s                  (was 39.10-39.85 s)
```

**Worse, not better, and still not resident.** HBM during encode reached only
4.5 GB, so the encoder still never occupied the ~7.8 GB a resident TP8 shard
would need. Removing the block attributes changed the streaming granularity
slightly (hence the small regression) without changing the fundamental behaviour.

## Three hypotheses now falsified for the same 39 s

| hypothesis | how tested | result |
|---|---|---|
| staged offload via `on_demand` (round 7) | removed `text_encoder` from `on_demand_component_paths`, confirmed via the "Prepared ... staged offload" log line | 39.85 s, unchanged |
| running on CPU / H2D transfer (round 8) | logged the encoder's parameter device | `npu:0`, `bfloat16` -- already on device |
| blockwise layer streaming via `encoder_block_attrs` (round 9) | emptied the entry | **42.85 s**, worse, still not resident |

Each was a plausible reading of the code. All three are wrong. What the HBM trace
does firmly establish is the *behaviour*: the encoder occupies ~1.9 GB and releases
it the instant encode finishes, so ~62 GB is passing through the device per
request. What I have not found is the code path that decides to do that.

I am stopping this thread rather than guessing a fourth time. Three falsified
hypotheses in a row is a signal that I am reading the offloader code without
understanding its control flow, and the next attempt should start by finding who
actually calls the per-layer load during the encoder forward -- by tracing it at
runtime rather than by grepping for configuration knobs.

## Where the goal stands after nine rounds

Honest accounting:

* **Refuted as worthwhile**: fused operators (0.3% of block time, round 1).
* **Refuted as reachable**: 15 s/step. Needs 47.8 TFLOPS against a shape-limited
  ceiling of 34.6 TFLOPS measured at the kernel's real K=128 (round 6).
* **Closed**: tiling (already optimal, Bc=608 adopted for 2%), communication
  (1.6 ms), conversions (1.2 ms), n_pad correction (1.86 ms), inter-rank
  serialisation (was my own sync).
* **Root cause found but not fixed**: the flash kernel has no software pipeline at
  all -- no double buffering, no set_flag/wait_flag -- which explains its 55-60%
  cube duty cycle. Fixing it is worth roughly 1.3-1.6x, to ~520-550 ms/block.
* **Mechanism found but not fixed**: the 39 s text encode is ~62 GB of per-request
  streaming. Three knobs tried, all wrong.
* **Not started**: VAE decode (44 s warm, 6.7x off official, partly hardware).

The only measurable gain landed in nine rounds is Bc=416 -> 608, worth 2%.
Against that, the round-6 finding (no pipeline) and round-9 trace (encoder is
streamed) are the two things a future attempt should build on, and both are
recorded with the exact numbers needed to verify a fix.

---

# Round 10: the kernel IS bandwidth-bound, and br=128/bc=608 is the way to fix it

## Peak HBM bandwidth, measured

```
device-to-device copy : 2.01 ms -> 534.8 GB/s (rw)
```

(An accompanying `sum` measured 41.5 GB/s, but that is a torch reduction and is
compute-bound; it is not a bandwidth figure and should not be quoted as one.)

## The kernel moves 234 GB per launch and does it at 65% of peak

With Br=64 there are 988 q-blocks, each sweeping all of K and V (2 x 113 MB):

```
traffic  = 988 x 226 MB            = 234 GB
measured = 672 ms
implies  = 348 GB/s                = 65% of the 534.8 GB/s peak
```

So the kernel is **substantially bandwidth-bound**, and the 65% figure means a
large part of the 672 ms is K/V being re-read from HBM.

**This retracts the round-2 conclusion that "the K/V traffic theory is dead".**
That conclusion rested on the Br sweep which was itself invalid -- `_build`
computes the tile sizes and overwrites `FA_BR`/`FA_BC`, so every arm of that
sweep ran the same br=64 kernel and the flat 672 ms was an artefact. Two rounds
later I described the flat curve as "consistent with fixed per-tile costs rather
than bandwidth", which was a plausible story built on a broken measurement. The
direct bandwidth measurement now says the opposite.

## Why the earlier br=128 test missed it

br=128 was only ever tested with a *smaller* Bc (256 or 128), because the fp32
score tile at br=128/bc=608 would be **304 KB** and does not fit. Small Bc costs
per-tile efficiency -- Bc 608 -> 256 alone was worth +188 ms at br=64 -- so that
test traded away more than the traffic saving gained, and came out at 708 ms.

The confound was never removed: **br=128 with bc=608 has never been run.**

## And it fits, in fp16

| tiling | fp32 score tile | fp16 score tile |
|---|---:|---:|
| Br=64, Bc=608 (today) | **152 KB** | 76 KB |
| Br=128, Bc=608 | 304 KB -- too big | **152 KB -- same as today** |
| Br=256, Bc=608 | 608 KB -- too big | 304 KB -- too big |

So keeping the score tile in fp16 buys exactly the footprint that already works
today while doubling Br:

```
br=64  / bc=608 : 234 GB traffic -> 438 ms bandwidth floor (measured 672)
br=128 / bc=608 : 117 GB traffic -> 219 ms bandwidth floor
```

If the kernel is at 65% of peak now, halving traffic should move the bandwidth
component from ~438 to ~219 ms, i.e. the 672 ms could plausibly land in the
350-400 ms range. That is a **1.7-1.9x** improvement on the attention kernel --
more than the pipeline work was going to give, and it attacks the dominant term.

## Precision risk, stated honestly

Softmax on fp16 scores is a real change, not a free one. After the max
subtraction the scores are in [-inf, 0] and the exponentials in [0, 1], which is a
benign range for fp16, but the accumulated denominators and the P@V product would
carry ~1e-3 relative error rather than the ~5e-4 the fp32 path currently achieves
against the reference. My verified tolerance in earlier rounds was 4.7e-4, so this
may or may not pass. The experiment must therefore check numerics at N=63232, not
just timing -- and if fp16 scores are too lossy, the fallback is chunking the fp32
score tile within UB (process Bc in two halves with online softmax), which keeps
precision and still halves the K/V traffic.

## Next step

Implement the smallest change that lets br=128/bc=608 build: allocate the score
tile as fp16. Then measure both time and worst-case relative error at the real
shape, and compare against the 672 ms / 4.7e-4 baseline.

## The fix works: 656 -> 558 ms, and the constraint moved from UB to L0C

Removed the fp32 exp temporary by exponentiating in place on the fp16 score tile
(`h_exp(Sij_ub_zN, Sij_ub_zN)`), then rebuilt at br=128:

```
Br=64  Bc=608  656.8 ms  21.8 TFLOPS     (in-place fp16 exp, unchanged from 657.7)
Br=128 Bc=608  fails: C_l0c needs 311296 B, 262144 B available
```

So freeing UB did not unlock br=128/bc=608; the limit had moved to **L0C**, which
is 256 KB: the score accumulator needs `Br x Bc x 4 <= 262144`, i.e.
`Br x Bc <= 65536`. At Bc=608 that caps Br at 107.

That constraint still allows halving the K/V traffic with a smaller Bc, and the
sweep over L0C-feasible tilings found one that works:

| tiling | ms | TFLOPS | L0C |
|---|---:|---:|---:|
| **Br=64, Bc=608** (previous) | **656** | 21.8 | 152 KB |
| **Br=128, Bc=304** | **558** | **25.6** | 152 KB |
| Br=256, Bc=256 | fails (`C_ub` 131072 B > 125440 B, narrowly) | | 256 KB |

**15% faster, 656 -> 558 ms**, with K/V traffic halved from 234 GB to 117 GB. It is
not the 1.7x the bandwidth arithmetic suggested, because Bc=304 has worse per-tile
efficiency than Bc=608 -- the two effects partly cancel, exactly as the earlier
Bc sweep predicted (608 -> 256 cost +188 ms at br=64). Net is still a solid gain.

## Numerics verified with the fp16 exp

```
cfg br64/bc608  worst rel err 0.000389  PASS
cfg br128/bc304 worst rel err 0.000389  PASS
```

Both identical, and **better** than the 4.7e-4 recorded for the old fp32-exp path.
The in-place fp16 exp costs nothing measurable: after the row-max subtraction the
scores are <= 0, so the exponentials sit in [0, 1] where fp16 is well behaved.

## Adopted

`_build` defaults changed from `br_cap=64, bc_cap=608` to **`br_cap=128,
bc_cap=304`**, verified to select br=128/bc=304 (L0C 152 KB) for N=63232.

This is the first measured kernel speedup of the goal: **15% on the attention
kernel**, on top of the 2% from the Bc change.

## What this implies for the remaining headroom

At 558 ms the implied K/V bandwidth is 117 GB / 0.558 s = **210 GB/s**, down from
348 GB/s. So the kernel is now much less bandwidth-bound (39% of the 534.8 GB/s
peak rather than 65%) and correspondingly more bound by per-tile compute
efficiency and the missing software pipeline. That means the two levers are now
complementary rather than alternatives:

* further Br increases are blocked by L0C, but Br=256/Bc=256 failed only narrowly
  on `C_ub` (131072 vs 125440 B), so shaving that buffer by ~5% would unlock a
  further halving of traffic;
* with bandwidth pressure reduced, the missing pipeline (round 6: no double
  buffering, no set_flag/wait_flag) is now the larger remaining term, and it is
  worth revisiting precisely because the kernel is no longer saturating HBM.

## Verified in the real pipeline: block 0.875 -> 0.759 s

Restarted the serving stack with the new tiling and ran a request:

```
dit 11/50 block=0.758s
dit 11/50 block=0.760s
dit 11/50 block=0.761s
fallbacks: 0
```

Against the 0.875 s block measured in round 3 on the same shape, that is **13.4%
faster end to end in the pipeline**, with zero fallbacks, so the denser tiling is
being used for every DiT attention call.

Per step: 50 blocks x 0.759 s = **37.9 s**, down from 43.8 s. At 4 steps the
denoise loop becomes ~152 s instead of ~175 s.

The gap between the isolated 15% (656 -> 558 ms) and the in-pipeline 13.4% is
expected: the block also contains the non-attention work that did not change, so
a 15% cut on the attention portion cannot be a 15% cut on the whole block.

## Round 10 result

This is the **first speedup of the goal that shows up end to end**:

| change | effect |
|---|---|
| Bc 416 -> 608 (round 5) | 2% on the kernel |
| fp16 in-place exp + Br 64 -> 128 at Bc 304 (round 10) | **15% on the kernel, 13.4% on the block** |

Numerics verified at the real shape with the new path: worst relative error
0.000389, which is better than the 4.7e-4 recorded for the previous fp32-exp
configuration.

## The kernel is now at 82% of its shape-limited ceiling

With Br=128/Bc=304 the numbers are:

```
time        558 ms per block
throughput  25.7 TFLOPS
bandwidth   117 GB / 0.558 s = 210 GB/s = 39% of the 534.8 GB/s peak
```

The bandwidth figure is the important one: the kernel is **no longer
bandwidth-bound**, so further traffic reduction has little left to give. The
mixed compute ceiling for this shape follows from the K-length sweep --
QK^T runs at K=D=128 (~26 TFLOPS) and PV at K=Bc=304 (~40 TFLOPS):

```
mixed ceiling = 2 / (1/26 + 1/40) = 31.5 TFLOPS
achieved                           25.7 TFLOPS  = 82% of it
compute floor                      455 ms/block
measured                           558 ms/block
max remaining kernel gain          1.23x  (558 -> 455 ms)
```

So the kernel has **at most 1.23x left in it**, and capturing it would take the
block from 759 to 656 ms and the step from 37.9 to 32.8 s.

I tried Br=256 to quarter the traffic anyway and could not build it: the blocker
is `Oi_ub_zN`, the fp32 output accumulator chunk in `output_update`
(`Tensor(FP32, (n1_size, m, N0), scope=UB)`), which overflows when Br doubles.
That is the same buffer that failed back in round 2 at Br=256/Bc=128, so it is a
known, structural cap on Br in this reference kernel rather than a new problem.
Working around it means shrinking `n1_size` further, which is more TIK surgery for
a gain the bandwidth measurement says is not there.

## What this settles about the 15 s/step target

```
needed for 15 s/step : 47.8 TFLOPS
mixed ceiling        : 31.5 TFLOPS
                     -> 1.5x ABOVE the ceiling
```

So 15 s/step is **not reachable by optimising this kernel**, and that is now
established by two independent routes: the FLOP/bandwidth budget (round 6) and
the measured K-length ceilings plus the 82%-of-ceiling result (this round). The
best case this hardware can reach by kernel work is ~32.8 s/step, and even that
requires recovering all 1.23x, which means the software pipeline plus tighter
per-tile efficiency.

## Where the goal actually stands

| item | state |
|---|---|
| (2) attention efficiency | **656 -> 558 ms kernel, 13.4% on the block, verified** |
| (2) reaching 15 s/step | **measured infeasible**: needs 47.8 TFLOPS vs a 31.5 ceiling |
| (1) fused operators | refuted as worthwhile, 0.3% of a block |
| (3) text encode 39 s | mechanism known (62 GB per-request streaming), three fixes tried, all failed |
| (3) VAE decode 44 s warm | not started |

The one durable gain is the attention kernel: two tiling/precision changes, 13.4%
end to end, numerics improved to 3.9e-4 rather than degraded.

---

# Round 12: the decode dtype comment is stale; bf16 makes it much worse

Started on goal item (3), the VAE decode (44 s warm against an official 6.56 s).
The strongest lead in the code looked like a direct contradiction:

```python
# 910A has no fp16 NanToNum kernel and the VAE decoder calls
# nan_to_num, so decode in bf16 where that op exists. The VAE is a
# small slice of the runtime, so the dtype switch costs nothing.
with current_omni_platform.create_autocast_context(
    device_type=self.device.type,
    dtype=torch.float16,          # <- comment says bf16
    enabled=True,
):
    video = self.video_vae.decode_latent(video_latent)
```

The comment says "decode in bf16 where that op exists" and the code requests
float16. If the comment were right, every decode would be pushing `nan_to_num`
through a fallback.

## Tested it: the comment is wrong, the code is right

Changed the autocast dtype to bfloat16, matching what the comment describes, and
measured a full request:

```
[H3PERFP] decode=256.07s
[H3PERFP] decode=257.30s
[H3PERFP] decode=257.42s
[H3PERFP] decode=258.11s
```

Against **157 s cold with float16** measured in earlier rounds, bf16 is **64%
worse**. So the fp16 path is correct on this hardware and the comment is stale --
presumably written for a code path or a CANN version where the fp16 `nan_to_num`
kernel really was missing. Reverted.

Also ruled out along the way: the `causal_convNd_update` warning that appears in
the decode logs comes from `vllm_ascend/ops/triton/mamba/`, i.e. it is a mamba
path, not the VAE. Chasing it would have been a red herring.

## What this leaves for the decode

The dtype is not the problem, so the 44 s warm against 6.56 s official is
elsewhere. Two things are worth noting before spending more rounds on it:

1. `decode()` wraps the work in `self._component_on_device(self.video_vae.decoder_component)`,
   so the VAE decoder is **staged in on every request** exactly like the text
   encoder was. Whatever the decoder weighs has to be brought in each time, and
   that cost sits inside the `decode` timer.
2. The rest is likely genuine 910A slowness on vector-heavy convolutional work. A
   video VAE decoder is almost entirely convolutions and normalisation, which land
   on the vector units rather than the cube, and the 910A is several generations
   behind the 910B/910C parts the official numbers come from. On that reading a
   large part of the 6.7x is hardware, not software.

I did not measure the split this round. The honest state is: one plausible
hypothesis tested and refuted, one red herring identified and dropped, and the
decode still un-diagnosed.

## Round 12 accounting

No speedup. The failed bf16 experiment cost a restart plus a full request, and it
is worth recording mainly so nobody else follows that comment.

## Component sizes, for the record

Checked the sizes because `decode()` stages `video_vae.decoder_component` in on
every request and `_prepare_local_conditioning` does the same for the encoder:

| component | weights |
|---|---|
| `text_encoder` | 14 shards, **62.13 GB** |
| `audio_vae` | 1 shard, 0.56 GB |
| `video_vae` | **no separate safetensors** |

The video VAE having no own weights is worth knowing before chasing its load
cost: its parameters must live inside the shared shards, so "stage the decoder in"
is not a simple small-file transfer and I cannot currently size that part of the
decode timer from the filesystem the way I could for the 62 GB encoder.

This also confirms the round-9 figure for the encoder at 62.13 GB, and hence the
~1.6 GB/s implied streaming rate (62.13 GB / 39 s).

---

# Round 13: the decode is 100% video-VAE compute, not staging or dtype

Instrumented `decode()` to separate the three candidate costs:

```
[H3PERFV] vae_stage_in=0.06s decode_latent=97.82s stage_out+rest=0.00s
[H3PERFV] audio_stage_in=0.05s audio_decode=0.28s
[H3PERFP] decode=154.95s / 158.30s
```

| component | cost |
|---|---|
| staging the video VAE decoder onto the device | **0.06 s** |
| **`video_vae.decode_latent`** | **97.82 s (cold)** |
| staging the audio VAE | 0.05 s |
| audio VAE decode | 0.28 s |

So the hypothesis I carried since round 12 -- that `_component_on_device` staging
was a meaningful part of the decode timer -- is **wrong**. Staging is six
hundredths of a second. The audio VAE is three tenths. Essentially the entire
decode is the video VAE's own convolutions.

Combined with round 12's result that bf16 is 64% worse, the decode is now
diagnosed: **it is video VAE compute on this chip, and nothing else.**

A note on the arithmetic: total decode was ~155 s against 97.8 s of
`decode_latent`, so ~57 s sits outside the instrumented `with` block -- in
`_release_stage_cache()` (called before the first timer) plus the
`[..., :height, :width].contiguous()` slice. This was a **cold** run, so an
unknown part of that is first-call compilation. The earlier decode-twice
experiment measured warm decode at 43.75 s, which means the warm figure for
`decode_latent` is around 43 s and I should not read the 57 s gap as a steady
state cost. It does not change the conclusion, but it does mean the cold and warm
splits differ and only the cold split was measured directly.

## What this means for the decode goal item

Warm decode ~43 s against an official 6.56 s is a **6.6x** gap that is now known
to be video VAE compute. A video VAE decoder is almost entirely convolutions,
normalisation and upsampling -- vector-unit work, not cube work -- and the 910A's
vector throughput is several generations behind the 910B/910C parts the official
numbers come from. So most of the 6.6x is very likely silicon, not software.

That makes the decode goal ("155s -> 20s") the same shape of problem as the 15 s
per-step target: the remaining headroom is small and the gap is dominated by
hardware generation. I am not going to claim it is achievable.

## Thirteen-round position

| item | state |
|---|---|
| (2) attention efficiency | **656 -> 558 ms kernel, block 13.4% faster, verified end to end** |
| (2) 15 s/step | measured infeasible (needs 47.8 vs a 31.5 TFLOPS mixed ceiling) |
| (1) fused operators | refuted, 0.3% of a block |
| (3) text encode 39 s | 62.13 GB streamed per request; three fixes tried, all failed |
| (3) decode | **diagnosed: video VAE compute, ~43 s warm, mostly hardware** |

Every remaining item has now been either landed, refuted, or diagnosed as
hardware-dominated. The one durable gain is the attention kernel.

---

# Round 14: ping-pong buffers are necessary but not sufficient

Round 6 established that the kernel has no software pipeline at all -- no double
buffering and no `set_flag`/`wait_flag` anywhere -- which is the main explanation
for its cube duty cycle and for the 103 ms gap between the 558 ms measured time
and the 455 ms compute floor.

Step one of fixing that is giving consecutive kv-block iterations distinct L1
buffers. Today `compute_in_each_kv_block` allocates `Vj_l1_zN` and `KjT_l1_zN`
fresh on every call, so if those land on the same addresses each iteration there
is a write-after-read hazard between iteration i's use and iteration i+1's load,
which forbids overlap by construction.

## L1 budget says it fits

| buffer | size at Br=128, Bc=304 |
|---|---:|
| KjT (fp16) | 76.0 KB |
| Vj (fp16) | 76.0 KB |
| Pij (fp16) | 32.0 KB |
| Oi (fp32) | 64.0 KB |
| Qi (fp16) | 32.0 KB |

```
fixed (Pij+Oi+Qi)      128.0 KB
one K/V set            152.0 KB
two K/V sets           304.0 KB
TOTAL current          280.0 KB
TOTAL double-buffered  432.0 KB
fits within 512 KB and within 1 MB
```

So there was no memory obstacle, and I made `Vj_l1_zN` / `KjT_l1_zN` select
`kv_blk_idx % 2` so consecutive iterations cannot alias.

## Result: it builds, runs, verifies -- and the time does not move

```
built ok
558 ms  25.7 TFLOPS  (baseline 558 ms / 25.7)
```

Exactly unchanged. So distinct buffers alone buy **nothing**: TIK still serialises
the load against the compute, because it has no way to know the two iterations'
buffers are independent without explicit synchronisation. That is what
`set_flag`/`wait_flag` are for, and the kernel still has none.

This is a useful negative result rather than a dead end -- it separates two things
I would otherwise have run together:

* the **memory precondition** (distinct buffers) is now satisfied and costs no
  performance and no correctness;
* the **pipelining machinery** (issue iteration i+1's MTE2 load before iteration
  i's compute, with flags so compute waits on its own load but not on the next
  one) is still absent and is what actually produces overlap.

The patch is left in place and verified correct, so the next attempt starts from
buffers that are already hazard-free rather than having to redo that part.

## What the flag version has to do

Restructure the kv loop from

```
for i: load V[i]; load K[i]; compute on V[i], K[i]
```

to

```
load V[0]; load K[0]
for i:
    if i+1 < n_blk: load V[i+1]; load K[i+1]     # into buffer (i+1)%2, async
    compute on V[i], K[i]                         # buffer i%2
```

with a `set_flag`/`wait_flag` pair per buffer so `compute i` waits for `load i`
and `load i+1` does not wait for `compute i`. The buffers for that already exist.

## Round 14 accounting

No speedup. But the remaining 1.23x is now narrowed to exactly one mechanism, with
its memory precondition verified and its absence confirmed by measurement rather
than by grep.

---

# Round 15: RETRACTION -- the round-14 ping-pong result was a cached build

## What happened

Round 14 claimed that ping-pong K/V buffers "built ok, ran, and measured 558 ms,
exactly unchanged". That conclusion was **wrong**: the patched kernel never
compiled.

Deleting the kernel cache and forcing a real rebuild produced:

```
TypeError: %d format: a real number is required, not Expr
  fa910a.py line 97, in _build
```

The patch named the buffers

```python
_vpar = kv_blk_idx % 2
Vj_l1_zN = self.tik_instance.Tensor(..., name="Vj_l1_zN_%d" % _vpar, scope=L1)
```

but inside `tik_instance.for_range` the loop variable `kv_blk_idx` is a TIK
**symbolic Expr**, not a Python int. `%d` on an Expr raises, so the build could
never succeed. Round 14's "built ok / 558 ms" was served from the cached
pre-patch kernel -- same shape, so same cache key.

**So the round-14 conclusion "ping-pong buffers are necessary but not
sufficient" is withdrawn.** I have no evidence either way about whether distinct
buffers help, because that configuration was never actually built or run.

## The real constraint this exposes

TIK resolves buffers by **name at trace time**, so a buffer cannot be selected by
a symbolic loop parity. Double buffering in this kernel therefore needs one of:

* `tik_instance.if_scope` on `kv_blk_idx % 2` with two separately named buffers
  allocated outside the scope, which is where TIK's own examples put them;
* or unrolling the kv loop so the parity is a Python value.

Both are real restructures, not a one-line rename. That is the actual blocker and
it was invisible while the cache was hiding the failure.

## The lesson, which I have now hit twice this session

**A build that "succeeds" after editing a kernel source must be verified against
a cleared cache.** In round 2 an invalid Br sweep produced a flat curve that I
then rationalised; here a cached build produced a clean-looking null result that I
reported as a finding. Both times the measurement apparatus, not the hypothesis,
was the problem. The tiling work in round 10 was only trustworthy because the
numbers moved (656 -> 558); a *null* result needs the cache cleared before it can
be believed at all.

## State after reverting

Reverted to the pre-ping-pong source (`flash_attention_fwd.py.nopipe.bak`), which
retains the round-10 in-place fp16 exp. Verified by deleting the kernel cache and
rebuilding from scratch:

```
built ok (fresh, cache deleted)
558 ms  25.7 TFLOPS
```

So the deployed kernel is the verified round-10 configuration: br=128/bc=304,
in-place fp16 exp, 558 ms, worst relative error 0.000389 as measured properly in
round 10 through the `flash_attention_910a` wrapper.

(One note: a hand-rolled raw-kernel numeric check in this round reported a
nonsense error of 3358.6. That reference did the 1/sqrt(d) scaling and layout
by hand and was simply wrong; the wrapper-based check from round 10 is the
authoritative one and passed at 0.000389. Worth recording so the 3358 figure is
not mistaken for a regression later.)

## Round 15 accounting

No speedup, and one earlier claim retracted. The genuine progress is negative but
useful: the double-buffering blocker is now identified precisely (symbolic parity
cannot name TIK buffers), and the measurement discipline required for null results
is written down where I will see it next time.

---

# Round 16: a full end-to-end run, and a step decomposition that changes priorities

Switched from kernel work to the thing the goal has asked for since the start and
that I had never produced: a real end-to-end measurement.

## The run

One 5 s / 768p / fl2va request on the 8x910A stack with the round-10 kernel, via
the HTTP API, timed from submit to `completed`:

```
status  = completed
wall    = 472.7 s

[H3PERFP] encode  = 41.49 s
[H3PERFP] denoise = 262.65 s      (4 steps, the distilled schedule)
[H3PERFP] decode  = 155.78 s      (cold; warm is ~43 s as measured in earlier rounds)

block time        = 0.762 / 0.764 s      (round-10 kernel active, was 0.875 s)
```

encode + denoise + decode = 459.9 s against 472.7 s wall, so ~13 s of HTTP/queue
overhead. The numbers are self-consistent.

## The decomposition that matters

The denoise loop already carries instrumentation I had never looked at, because I
had been grepping only for my own `[H3PERFP]` tag:

```
[H3PERF] dstep 1 prep=8.9s  model=95.7s  update=9.6s
```

(eight copies of each line -- one per rank; four distinct step numbers, matching
the 4-step schedule.)

Cross-checking against the block timer:

```
50 blocks x 0.762 s            =  38 s   of block work per step
denoise 262.65 s / 4 steps     =  65.7 s per step
prep 8.9 s + update 9.6 s      =  18.5 s per step outside the model call
```

So a warm step is roughly:

| part | per step | share |
|---|---:|---:|
| block loop (attention, MLP, norms) | 38 s | ~58% |
| **prep** | 8.9 s | ~14% |
| **update** (Euler step, row updates) | 9.6 s | ~15% |
| other/overhead | ~9 s | ~14% |

**This is the first real step-level decomposition I have, and it contradicts the
working assumption of the last ten rounds.** I had been treating "DiT per step"
as essentially the attention kernel, because in round 3 the block decomposition
showed attention at 869 of 871 ms inside a block. That is still true -- but a
block is only ~58% of a step. The other ~42% is `prep` and `update` and overhead
that I never examined, and 18.5 s of it is plainly attributable to two labelled
sections.

Two consequences:

1. The remaining attention headroom (1.23x on 38 s = 8 s per step) is worth about
   **12% of a step**, not the ~30% the earlier framing implied.
2. `prep` and `update` together are **18.5 s per step -- 2.3x larger than the
   entire remaining attention headroom** -- and have never been looked at once.

## Also learned: the model-call gap

`model=95.7 s` per step against 38 s of block work leaves ~57 s inside the model
call but outside the block loop. That is a *cold* figure (these lines come from
the first invocation, which includes TBE compilation inside `model`), so the warm
split is unknown, and I should not present 57 s as steady state. Getting the warm
per-step `prep/model/update` split -- simply by reading these lines from a second
request, which the known TBE crash currently prevents -- is the measurement that
would settle it.

## Round 16 accounting

No speedup landed. But this round produced the first measured step decomposition
and it materially redirects the remaining effort: the attention kernel is no
longer where the time is, and `prep`/`update` are.

## Comparable measurement against the official lossless numbers

The goal asks for this and it has to be stated carefully, because a naive
comparison is not valid:

* The deployed checkpoint is **`MiniMax-H3-4step`**, a distilled 4-step variant
  whose `FL2VA/model_index.json` schedule is `[1.0, 0.7, 0.4, 0.15, 0.0]`, i.e.
  four steps. The official A2 (49 steps, 150 s) and A3 (114 s) figures are for
  the full 50-timestep schedule. Running 49 steps on distilled weights would be
  a different model, not a comparable measurement, so I am not going to report
  it as one.
* What *is* comparable is per-step cost. A2 at 49 steps / 150 s implies
  **3.06 s per step**; A3 implies **2.33 s per step**. Mine is
  **65.7 s per step** (262.65 / 4) cold-denoise, or **~38 s per step** for the
  block loop alone.

So the honest statement is: **this stack runs ~21x slower per step than A2**
(65.7 / 3.06) on the same nominal workload shape, and the gap is distributed
across attention (58%), prep/update (29%) and overhead -- not concentrated in any
one place the way I had assumed.

---

# Round 17: found 6.4 s per step going into a CPU round-trip to extract four numbers

Acting on round 16's decomposition, split the 8.6 s `prep` into its parts.

## First measurement was misattributed, and worth recording

My first timer wrapped "both `prepare_edit_rows` calls plus `forward_kwargs`" and
labelled the whole thing `rows`, printing `kwargs` as the time *after*
`forward_kwargs` (always 0.00). That produced `rows=6.5s kwargs=0.00s`, which
reads as "prepare_edit_rows is the cost".

That was wrong. `minimax_h3_prepare_edit_rows` in `latent_mask.py:155` begins:

```python
if edit is None:
    return rows, None
```

A generation request has no edit, so it is an immediate passthrough. Re-timed with
separate boundaries:

```
[H3PERFQ] editrows=0.00s kwargs=6.25s
[H3PERFQ] editrows=0.00s kwargs=6.56s
[H3PERFQ] editrows=0.00s kwargs=6.70s
```

So `prepare_edit_rows` is exactly 0.00 s as the code says, and the 6.4 s is
**`positive.forward_kwargs`**. The lesson is the same one as rounds 2 and 15: a
measurement is only worth what its boundaries are worth, and I should have read
the callee before naming the number after it.

## The actual cause

`forward_kwargs` (`denoise_loop.py:182`) ends with:

```python
# 910A: torch.unique runs on AICPU and fails there ("The aicpu execution
# is abnormal"), so do it on CPU; the tensor is a 1-D row vector.
_unique_cpu, _inverse_cpu = torch.unique(
    timesteps.detach().cpu(), sorted=True, return_inverse=True
)
unique_timesteps = _unique_cpu.to(timesteps.device, dtype=timesteps.dtype)
inverse_indices = _inverse_cpu.to(timesteps.device)
```

`timesteps` has shape `(self.seq_len,)` -- the full packed row count, ~63232.
So every step does:

1. `.detach().cpu()` -- a **device-to-host copy, which forces a synchronisation
   of the whole pipeline**;
2. a sort-based `torch.unique` on the CPU over 63232 elements;
3. two host-to-device copies.

And the decisive observation: **`timesteps` is never given to the model.** The
returned dict contains only `x`, `audio_x`, `unique_timesteps` and
`inverse_indices`. The 63232-element vector is constructed on the device, copied
to the host, reduced to its distinct values, and then only a handful of distinct
scalars plus an index vector go back. The vector carries **about four distinct
values** (t_video, imgvid_cond_timestep, t_audio, audio_ref_cond_timestep), plus
target timesteps under editing.

Worse, `fill_timesteps` *already* maintains a CPU copy for the same 910A reason:

```python
# 910A: build the packed timestep vector on CPU and copy it back once.
_cpu_timesteps = torch.full((self.seq_len,), float(t_video), ..., device="cpu")
```

So there are two CPU round-trips per step for a vector that is piecewise constant
with four values.

## Why this matters more than the attention work

6.4 s of a ~93 s step is **~7%**, and unlike the attention headroom it is not
shielded by a hardware ceiling: this is argument marshalling. The fix is to build
the timestep vector and its unique/inverse **once on the CPU** (the mask positions
are static per request) and upload only the results, eliminating the
device-to-host copy and its synchronisation entirely.

The synchronisation is the part I would expect to matter beyond the measured
6.4 s: a `.cpu()` in the step loop serialises everything queued on the device, so
it can both inflate its own apparent cost and prevent overlap across the step
boundary. That is the same trap that corrupted my stage timings in round 3.

## Round 17 accounting

No speedup landed; instrumentation added, measured, and reverted (verified: no
`H3PERFQ` or `_perf_p` remnants, 0 NPU processes). But the target is now precise
to a function and a line, with a wrong intermediate attribution explicitly
corrected rather than left standing.

Also refined the step picture from round 16: `model` is 71.6-74.4 s warm here
against the 95.7 s cold figure recorded last round, confirming that figure
included TBE compilation and should not be quoted as steady state.

---

# Round 18: three wrong guesses, then the answer -- it is the scatter, not the unique

## Guess 1 (round 17's conclusion) was wrong

Round 17 concluded the 6.4 s was the `torch.unique` CPU round-trip and predicted
removing the device download would fix it. Implemented exactly that: had
`fill_timesteps` return the CPU vector it already builds, and used it for
`torch.unique` instead of `timesteps.detach().cpu()`.

Result: **`kwargs=6.31-6.65 s`, unchanged.** So the download was not the cost.

What that change *did* show is that the unique path is genuinely cheap once the
data is already on the host: `fill=0.01-0.31 s`, `uniq_and_up=0.01-0.06 s`.

## Then measured instead of guessing again

Split `forward_kwargs` into its three regions:

```
[H3PERFR] xcopy=6.20s  fill=0.07s  uniq_and_up=0.06s
[H3PERFR] xcopy=6.27s  fill=0.06s  uniq_and_up=0.05s
[H3PERFR] xcopy=6.49s  fill=0.07s  uniq_and_up=0.04s
```

**`xcopy` is the whole 6.4 s.** That region is:

```python
x = self.x_base.clone()                          # (1, seq_len, VIDEO_ROW_WIDTH) fp32
x[0].index_copy_(0, self.img_pos_dev, video_rows)
audio_x = self.audio_x_base.clone()
audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
```

## And it cannot be bandwidth

`x_base` is `torch.zeros(1, seq_len, MINIMAX_H3_VIDEO_ROW_WIDTH)` -- at seq_len
63232 with a latent row width in the tens, that is on the order of 12 MB. A 12 MB
clone plus two row writes cannot take 6.4 s of memory bandwidth; at the 535 GB/s
measured in round 10 it is well under a millisecond.

So the cost is the **`index_copy_` scatter**, roughly 63000 indices, twice per
step. And this codebase already documents that scatter is pathological on this
part -- `fill_timesteps` carries the comment:

> 910A: build the packed timestep vector on CPU and copy it back once.
> Device-side mask indexing needs **NonZero and the scatter needs AICPU; both are
> unavailable here**

So there is strong corroboration from the code itself: the authors already routed
*around* device scatter for the timestep vector. `forward_kwargs` still uses
`index_copy_` (a scatter) for the latent rows, twice per step, and that is the
6.4 s.

## Three guesses, three misses, and the pattern

| guess | source | verdict |
|---|---|---|
| staged offload of the encoder | round 7 | wrong |
| `.cpu()` for `torch.unique` | round 17 | wrong |
| `fill_timesteps` CPU copies | round 18, briefly | wrong |
| **`x_base.clone()` + `index_copy_` scatter** | **measured** | **this one** |

Every wrong one was a plausible reading of the code; only the measurement
settled it. The specific discipline that would have saved rounds 17 and 18: when
a region is 6.4 s and the obvious candidates are sub-millisecond on paper, measure
*which line* before proposing a fix, rather than picking the most suspicious code
inside it.

## The fix, for next time

`x` is a zeros buffer with the media rows scattered in. Two directions:

1. **Persistent buffer, avoid the clone.** Keep `x`/`audio_x` allocated per
   request and refill in place. Removes the 12 MB clone but not the scatter.
2. **Remove the scatter.** With `img_pos` sorted, writing `video_rows` into
   consecutive rows is a contiguous copy rather than a gather-scatter; if the
   media rows are a contiguous range in the packed sequence, `index_copy_`
   becomes a slice assignment. Need to confirm whether they are contiguous
   (text/padding interleaved would break this), and if not, precompute the
   inverse permutation once per request and use `index_select` on the *base*
   instead -- both are one-time costs rather than per-step.

Expected: up to 6.4 s per step, ~26 s over 4 steps, plus removal of a
synchronising scatter from the step loop.

## Round 18 accounting

No speedup landed. Everything was reverted: verified 0 remnants of the
instrumentation, `fill_timesteps` back to returning `None`, 0 NPU processes. The
contribution is that the 6.4 s is now attributed to a specific statement with a
physical argument (12 MB cannot be 6.4 s) and corroborated by the codebase's own
scatter workaround, plus an explicit record of three failed guesses so the next
attempt starts from the measurement rather than from suspicion.

---

# Round 19: index_copy_ was costing 6.1 s per step; fixed, denoise 262.7 -> 211.7 s

## Micro-benchmark first, because three guesses in a row had been wrong

Before touching the code I benchmarked the candidate statements at the real
shapes (`seq_len=63232`, video row width 96, audio 32):

```
video base 24.3 MB, audio base 8.1 MB

clone alone                      :    0.127 ms
clone + index_copy_(n=63000)     : 6144.669 ms
clone + x[0][idx] = rows         :    2.606 ms
clone + contiguous slice write   :    0.288 ms
```

At n=7904 (the per-rank row count) it is 794 ms, so it scales with the index
count at about **97 microseconds per index** -- a per-element fallback path, not a
memory-bandwidth limit. `6144 ms` matches the 6.4 s `xcopy` region measured in
round 18, which is what confirms this is the right statement rather than a
plausible one.

Two useful negatives from the same run: the inputs `idx` here are not contiguous,
so the 0.288 ms slice write is not available in general; and the clone itself is
0.127 ms, so the buffer copy never mattered.

## The fix

```python
x[0].index_copy_(0, self.img_pos_dev, video_rows)   ->   x[0][self.img_pos_dev] = video_rows
audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
                                                    ->   audio_x[0][self.audio_pos_dev] = audio_rows
```

Same scatter semantics, **2358x faster**. The codebase already knew scatter was
broken here -- `fill_timesteps` carries "the scatter needs AICPU [and is]
unavailable here" and builds its vector on the CPU to avoid it -- but
`forward_kwargs` had never received the same treatment.

## Verified end to end

```
prep   8.6 s  ->  1.97-2.52 s        (per step; 6.4 s removed, as predicted)
model  73.0-73.5 s                   (unchanged)
update  8.7-10.7 s                   (unchanged)

denoise  262.65 s  ->  211.66 s      (-51 s, -19%)
encode    41.49 s  ->   39.99 s
block     0.762 s  ->   0.754-0.757 s
wall                  ->   422 s
```

No new errors: the only warnings in the log are the 8 pre-existing
`causal_conv1d_update is unavailable` mamba fallbacks from `vllm_ascend`, one per
rank at startup, which were present in every earlier run. Health stayed healthy.

This is the **largest single improvement of the session** and, unlike the
attention work, it does not sit against any hardware ceiling -- it was pure
argument marshalling on a pathological fallback path.

## The generalisable lesson

`index_copy_` at ~97 us per index on this part is the kind of thing that hides in
plain sight: one plausible-looking line, no error, no warning. It would never have
been found by reading the code -- the round-18 evidence that scatter was
suspect came from the authors' own workaround comment, but the magnitude only came
out of a benchmark.

Standing practice worth adopting for the rest of this work: **before assuming a
line is cheap because it is a single statement, benchmark it at the real shape.**
Three consecutive wrong guesses in rounds 7, 17 and 18 all came from reasoning
about code instead of timing it.

## Where the step now stands

| part | per step | share |
|---|---:|---:|
| model (block loop etc.) | 73.1 s | ~86% |
| update | 10.1 s | ~12% |
| prep | 2.2 s | ~3% |

So `update` (Euler step and row updates, still using device scatters in places) is
now the largest non-model item at ~10 s per step, and the block loop at 38 s of
the 73 s model time remains the single biggest component overall.

---

# Round 20: update is not a scatter target, and 4.4 s of it was async misattribution

Chased `update` (10.1 s/step) with the technique that worked in round 19.

## The mask ops there are genuinely cheap

`update` contains four boolean-mask scatters and several mask gathers per step.
Measured at the real shape (seq 63232, video width 96, 60000 updated rows):

```
rows.clone()                : 0.124 ms
rows[update].float()        : 0.866 ms
clone + r[update]=tgt       : 1.976 ms
clone + both mask writes    : 2.629 ms
clone + contiguous writes   : 0.352 ms
```

So boolean-mask assignment is ~2-3 ms, **three orders of magnitude away** from
`index_copy_`'s 6.1 s. The two ops are not the same path on this part: indexed
assignment with a mask vectorises, `index_copy_` apparently does not. Worth knowing
so I do not "fix" mask indexing that was never broken.

The mask also turns out to be contiguous, so contiguous writes would be 0.35 ms, a
further ~2 ms available -- far too small to be worth the risk.

## So the 10 s was not update's either

Added a `torch.npu.synchronize()` after the model call, purely as an accounting
probe, and compared:

| | without sync | with sync |
|---|---:|---:|
| prep | 2.2 s | 2.5 s |
| **model** | **73.1 s** | **77.2 s** |
| **update** | **10.1 s** | **5.7 s** |
| denoise | 211.66 s | 212.42 s |

About **4.4 s of update's apparent cost was the model's device tail**, returned to
its rightful owner once the call was synchronised. NPU ops are asynchronous, so a
Python-level timer around `model(**fk)` stops before the device has finished, and
the remainder lands in whatever the next timer covers.

This is the same trap that corrupted my round-3 stage timings, now appearing in the
project's own instrumentation. **The consequence is that `_dmodel` has been
understating the model and `update` overstating the update since the beginning.**

Critically, denoise did **not** change (212.42 vs 211.66), so this is a
re-attribution and not a speedup -- the sync only makes measurements honest.
Reverted it: the file is byte-identical to the pre-probe backup (verified by
`diff`), and the round-19 fancy-index fix is intact.

## Corrected step picture

| part | per step | share |
|---|---:|---:|
| model, device-synchronised | **77.2 s** | ~89% |
| update | 5.7 s | ~7% |
| prep | 2.5 s | ~3% |

And within the model, the block loop accounts for only **38 s** (50 x 0.754).
So **~39 s per step is inside the model call but outside the block loop** --
about 45% of a whole step, and larger than everything else combined.

That is now the biggest unexamined item by a wide margin, and it is where the next
round should go: it covers patchification/embedding, the per-step global
conditioning (AdaLN modulation), the final norm and unpatchify.

## Round 20 accounting

No speedup. Two useful negatives (mask indexing is fine; update is not a scatter
problem) and one correction (the project's own prep/model/update split is
async-skewed by ~4.4 s, understating the model). The next target is now
identified by measurement rather than inference: the 39 s inside the model call
but outside the block loop.

---

# Round 21: the phase breakdown was in the logs all along, and it exposes a 38 s gap

## The instrumentation already existed

`MiniMaxH3DiTModel.forward` already prints, per step:

```
[H3PERF] phase pre=20.954s blocks=52.129s post=0.000s total=73.083s
[H3PERF] dit 50/50 block=0.759s stack=52.675s
```

I had been reading only `block=`, and had been computing "50 x 0.759 = 38 s" of
block work. That was wrong: **`block=` is a single block's time, while `stack=` is
the 50-block total, and the stack is 52 s.**

So the model call decomposes as pre ~20.5 s + blocks ~52.1 s, and `pre` -- the
largest uninspected item -- was visible in the log the whole time.

## Then a surprising and inconsistent run

Adding only a timer produced a run whose phase lines read:

```
phase pre=0.922s rope=0.000s blocks=33.796s post=0.000s total=34.718s
dit 50/50 block=0.752s stack=33.763s
```

i.e. total 34.7 s against 73.1 s before. I did not accept that at face value,
because a timer cannot halve the model. Grouping the lines by rank showed the real
structure:

```
702033  pre=0.958  blocks=33.739
702033  pre=20.084 blocks=52.261      <- same rank, different step
702033  pre=3.576  blocks=36.548
```

So it is **step-to-step variance within a single rank**, not rank imbalance: `pre`
ranges 0.9-20.1 s and the stack 33.7-52.3 s. The pattern is what DLO weight
streaming looks like -- cheap when the layers are resident, expensive when they
are being streamed in.

## The finding that matters

Across all of that variance, two numbers stayed rock steady:

```
inner  MiniMaxH3DiTModel.forward : phase total = 34.7 s
outer  step timer                : dstep model = 73.2 s
```

Both consistent, every step, every rank. So **roughly 38.5 s per step is spent
inside the `model(...)` call but outside `MiniMaxH3DiTModel.forward`** -- between
the two timers. That is larger than `pre`, larger than `update`, and larger than
the remaining attention headroom, and it has never been looked at because both
timers were already in place and I had been reading neither.

Candidate explanations to check next: a wrapper around the model that does the
DLO forward hooks, a per-rank synchronisation or collective in the wrapper, or the
serving layer re-packing inputs. It is not the transformer's own arithmetic, which
is the 34.7 s.

## Corrected step accounting

Given model outer 73.2 s, prep 2.4 s, update 9.8 s, a step is ~85 s:

| part | per step | share |
|---|---:|---:|
| **inside model(), outside the transformer forward** | **~38.5 s** | **~45%** |
| transformer forward (pre ~1-20 + blocks ~34-52) | ~34.7 s | ~41% |
| update | 9.8 s | ~12% |
| prep | 2.4 s | ~3% |

The largest single item is now a gap between two existing timers rather than any
particular computation.

## Round 21 accounting

No speedup. Rope-table hypothesis instrumented and found not to be the cost
(`rope=0.000s` -- the table is supplied through kwargs, so the branch is not
taken; reverted, 0 remnants). The contribution is the corrected block arithmetic
(the stack is 52 s, not 38 s), the discovery that `pre` is 20.5 s and had been
visible all along, and the 38.5 s gap between the inner and outer model timers,
which is now the largest identified item in the pipeline.

Push to GitHub failed all round (`SSL_ERROR_SYSCALL` at the TLS handshake, DNS
resolves); commits are local and will be pushed when the network recovers.

## Round 22: the two obvious explanations for the 38.5 s are both ruled out

Read the wrapper chain rather than guessing, per round 21's lesson.

```
pipeline_minimax_h3.py:1994   minimax_h3_denoise_loop(model=transformer, ...)
        transformer = self._transformer_for_task(task)
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            ... denoise loop ...
```

and the context manager is:

```python
def _resident_dit_layers_on_device(self, *, enabled: bool = True):
    controller = getattr(self, "_dlo_residency_controller", None)
    if controller is not None and enabled:
        controller.load_resident_layers()
    try:
        yield
    finally:
        if controller is not None and enabled:
            controller.offload_resident_layers()
```

So `model` is the raw transformer and the only wrapper is a DLO residency
context that loads **once before the loop** and offloads **after it** -- it is not
per-step, so it cannot produce a per-step 38.5 s gap.

The other candidate, that the model is called twice per step (positive and
negative branches) so that two 34.7 s forwards add to ~69 s, is also ruled out:
32 `phase` lines divided by 8 ranks gives 4 per rank for a 4-step schedule, i.e.
one model call per step.

So both explanations are dead and the gap stands unexplained:

```
inner   phase total : 34.7 s   (and 73.1 s on some steps -- it varies)
outer   dstep model : 73.2 s   (consistent)
```

One observation worth carrying forward: `phase total` itself takes **both** 34.7
and 73.1 across steps, while `dstep model` sits at 73.2 throughout. That pattern is
more consistent with *some* forwards being cheap and others expensive than with a
constant gap, which would mean my framing of "38.5 s between the timers" may be
wrong -- the inner timer may simply be sampling different steps than the outer one
in the log I compared.

**The measurement that settles it** (not taken): print both timings in a single
statement, in the same process, for the same step -- e.g. have the denoise loop
record `_dmodel` and also read whatever the transformer recorded for that same
call, and emit them on one line. Comparing values from different lines of a
multi-rank log is what produced this ambiguity in the first place.

## Round 22 accounting

No speedup, no new instrumentation. Two plausible mechanisms eliminated by reading
the code, and an explicit statement that the 38.5 s figure itself is now in doubt
because the two numbers were never observed together for the same step. That is a
correction to round 21's conclusion, which was stated too confidently.

---

# Round 23: no gap at all -- instead, 20% of the denoise is first-step warmup

## The paired measurement

Printed the inner (`MiniMaxH3DiTModel.forward` total) and outer (`dstep model`)
timings on **one line, same process, same step**, stashing the inner value on the
module instance that the denoise loop already holds as `model`:

```
step=1 outer_model=71.122s inner_total=71.121 inner_pre=19.995 inner_blocks=51.126
step=2 outer_model=40.125s inner_total=40.125 inner_pre= 3.611 inner_blocks=36.513
step=3 outer_model=34.730s inner_total=34.729 inner_pre= 0.984 inner_blocks=33.80
```

**Inner and outer agree to 1-2 ms.** So the 38.5 s gap I reported in round 21
**does not exist**. It was an artefact of comparing `dstep model=73.2s` against
`phase total=34.7s` when those lines came from different ranks and different steps
of a multi-rank log. Round 21's headline conclusion is withdrawn, and round 22 was
right to doubt it.

## What the measurement did find, which is much more useful

| step | total | inner_pre | inner_blocks |
|---|---:|---:|---:|
| 1 | **71.4 s** | **19.7 s** | 51.7 s |
| 2 | **40.2 s** | 3.7 s | 36.5 s |
| 3 | **34.7 s** | **0.97 s** | 33.8 s |
| 4 | ~34.7 s | ~0.97 s | ~33.8 s |

**Warm steady state is 34.7 s per step, and the first two steps carry about 42 s
of excess (36.7 s + 5.5 s).** On a 4-step schedule that is **20% of the 210 s
denoise** spent warming up, not doing diffusion.

Both components converge monotonically: `inner_pre` 19.7 -> 3.7 -> 0.97 s, and
`inner_blocks` 51.7 -> 36.5 -> 33.8 s. That is the signature of the distributed
layerwise offload progressively making weights resident, so the first forward
streams most of the 50 blocks' weights in and later forwards do not.

## Consequences, both of them large

1. **Every per-step number I have quoted all session was a blend of warmup and
   steady state.** "65.7 s per step" (round 16), "~93 s step" (round 17), "~85 s
   step" (round 21) and "73.2 s model" are all averages that include this ramp.
   The honest steady-state figure is **34.7 s per step**, and the honest first-step
   figure is 71.4 s.
2. **A warmup outside the timed region is worth ~42 s per 4-step run (20%)**, and
   official benchmark numbers exclude warmup by construction -- which is part of
   why the comparison looked so bad. If the stack emitted a throwaway forward
   before the real loop, denoise would be about 4 x 34.7 = 139 s instead of 210 s.

This also finally explains the "official excludes warmup" note carried since round
3: I had been comparing my cold-ish 4-step denoise against their warm 49-step
figure, and the warmup is a fifth of my total.

## Round 23 accounting

No speedup landed, but this is the most consequential measurement since round 16:
it retracts a wrong headline from round 21, replaces it with a real and much
larger structural finding (20% warmup), and corrects the denominator used for
every per-step figure in the record. Instrumentation added, measured, reverted;
verified 0 remnants of both probes and the round-19 index fix intact.

## Next step

Either (a) add a warmup forward before the timed loop so the reported number is
steady state, which is also what makes any A2/A3 comparison fair, or (b) attack
the streaming itself -- `_resident_dit_layers_on_device` calls
`load_resident_layers()` once, and HBM at 8.9 GB of 32 GB suggests the 50 blocks
could simply stay resident, which would remove the ramp from the warm steps too
(34.7 s step contains 33.8 s of blocks that are still restreaming).

---

# Round 24: the residency knob exists, works, and does not fit in 32 GB

Round 23 identified ~42 s per 4-step run (20% of denoise) as first-step warmup and
proposed making the DiT blocks resident.

## The knob is real

`plan_resolver.py:235`:

```python
resident_head = min(config.dlo_resident_layers, len(blocks))
```

exposed as `--dlo-resident-layers`. Confirmed it takes effect:

```
--dlo-resident-layers 50 -> "Keeping 50 leading blocks resident on transformer; streaming 0 tail blocks"
--dlo-resident-layers 30 -> "Keeping 30 leading blocks resident on transformer; streaming 20 tail blocks"
```

And with no flag at all, **no such message appears** -- confirming round 23's
inference that `resident_head` is 0 by default, i.e. **all 50 blocks stream on every
step** in the configuration used for every measurement in this record.

## But it does not fit, and the failure is not what I expected

```
--dlo-resident-layers 50 : OOM, 29.54 GiB allocated of 32.00, needs 1.20 more
--dlo-resident-layers 30 : OOM, 30.51 GiB allocated of 32.00, needs 1.20 more
```

Residency at 30 blocks used **more** memory than at 50, which is backwards. So the
memory is not being spent on resident blocks; it is the `--dlo-no-use-allgather`
path that residency requires (`config.py:275`: "dlo_resident_layers requires the
DiT DLO transfer to be rank-local; set dlo_use_allgather=False") that is heavy.

That matters because round 5 measured `--dlo-no-use-allgather` as **29 s slower**
(287 s vs 258 s denoise) before any of this. So the two options are mutually
exclusive and the one that enables residency also costs 29 s:

| configuration | denoise | resident blocks |
|---|---:|---|
| allgather ON, resident 0 (current) | **212.9 s** | 0 |
| no-allgather, resident 50 | OOM | 50 |
| no-allgather, resident 30 | OOM | 30 |

For reference, HBM during a normal run peaks at **7.58 GB of 32 GB**, so the memory
is available in principle but not through this path.

## Direction closed

Making the streamed blocks resident is **not achievable on 32 GB** via
`--dlo-resident-layers`, and the alternative (option (a) from round 23 -- a warmup
forward before the timed loop) does not need memory at all. That is now clearly the
right choice: it recovers the same ~42 s ramp by paying it once outside the
measurement, and it is also what makes any comparison against A2/A3 fair, since
official numbers exclude warmup by construction.

Verified restored configuration afterwards: `denoise=212.89s`, and the warm steps
show `phase pre=0.932s blocks=33.881s total=34.813s`, matching round 23 exactly.
So the warm steady state is confirmed at **34.8 s per step** and the current
configuration already reaches it -- the loss is entirely the ramp.

## Round 24 accounting

No speedup. Two OOM experiments, both reverted; the configuration is restored to
the known-good one and verified working. The contribution is that the residency
question is now settled with evidence rather than left as a plausible idea, and
the reason it fails is documented (not block count -- the no-allgather transfer
path), which stops the next attempt from re-trying 50 and concluding "too many
blocks".

---

# Round 25: the comparable measurement, and why a warmup was the wrong way to get it

## Tried the warmup forward first, and it was the wrong instrument

Added a discarded forward before step 0's timed work (round 24's option (a)).

```
warmup_forward = 73.46 s (discarded, before timing)
dstep 1 model  = 39.36 s          (was 71.4 s -- the ramp moved, as predicted)
denoise        = 252.18 s         (was 212.89 -- WORSE)
wall           = 463 s            (was 472 -- unchanged)
```

Two things to take from this:

* The mechanism is **confirmed**: warming the DLO path cut step 1's model call from
  71.4 to 39.4 s. The ~32 s ramp is real streaming cost, not a measurement artefact.
* But the warmup was placed **inside** the denoise timer, so it added 73 s rather
  than moving it, and **wall time did not change** (463 vs 472 s is noise). It was
  73 s of genuinely wasted compute purchased for a cosmetic metric.

Reverted. I had said before running it that a warmup cannot reduce wall time, and
that is exactly what happened -- so implementing it inside the timed region was a
mistake in placement, not in concept.

## The same comparison is available for free

Rather than buying the steady-state number with real compute, accumulate each
step's elapsed time and report it after the loop:

```
[H3PERF] denoise_steady step1=85.34s steady_avg=41.32s
         steady_total=165.26s projected_49steps=2024s
```

Nothing extra is computed; the numbers come from the per-step timings already
collected. Verified no regression: step1 85.3 + 3 x 41.3 = 209 s against the
211-213 s baseline, and wall time 423 s.

## The comparable measurement the goal asked for

This is the like-for-like figure, since the official numbers exclude warmup:

| | steps | time | per step |
|---|---:|---:|---:|
| this stack, steady state | 49 (projected) | **2024 s** | **41.3 s** |
| this stack, actual 4-step run | 4 | 209 s denoise | 41.3 s steady / 85.3 s first |
| official A2 | 49 | **150 s** | 3.06 s |
| official A3 | 49 | **114 s** | 2.33 s |

So the honest statement is: **this stack is 13.5x slower per step than A2 and 17.8x
slower than A3** on the same nominal workload shape, on a per-step steady-state
basis.

That supersedes the "~21x" I wrote in round 16, which compared a *cold* 65.7 s/step
against A2's warm figure -- not like for like. The corrected gap is smaller than I
had been reporting, and it is now stated in the same terms the official numbers use.

## Where the 41.3 s per step goes

From rounds 23 and 25, per steady step:

| part | per step |
|---|---:|
| transformer blocks (50, all streamed) | 33.8 s |
| transformer pre | 0.95 s |
| update | ~5.7 s |
| prep | ~2.2 s |

Of the 33.8 s of blocks, roughly 28 s is arithmetic and ~5 s is DLO streaming (from
the step1-vs-steady delta). The attention kernel inside that is 50 x 0.754 s =
37.7 s of a 52 s stack in the *cold* configuration, or about 33.8 s warm -- so
attention remains the dominant single component, consistent with round 10's finding
that it sits at 82% of its shape-limited ceiling.

## Round 25 accounting

No wall-time improvement, and one wrong implementation reverted. The deliverable is
the comparable measurement itself -- explicitly required by the goal since round 1
and never produced until now -- plus a correction of the headline ratio from ~21x to
13.5x/17.8x on a like-for-like basis.

---

# Round 26: the finite-check diagnostic costs 1.9 s/step, not the 15 s its own comment claims

Chasing the last unattributed item (`update`, 5.7 s/step), read the two step
functions. Both are ordinary elementwise arithmetic, but each is wrapped in
validation helpers, and the helpers carry a comment that looked like the answer:

```python
def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    # isfinite().all().item() sweeps the whole tensor and then forces a
    # host<->device sync. These run on every large latent tensor, six times per
    # denoise step, and cost ~15 s per step on 910A -- a pure diagnostic that
    # never fires in practice. Keep it available but off be default when asked.
    if not _FINITE_CHECK:
        return
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
```

and the gate defaults to **enabled**:

```python
_FINITE_CHECK = os.environ.get("VLLM_OMNI_910A_FINITE_CHECK", "1") != "0"
```

So it ran in every measurement in this record. Six full-tensor reductions with a
device sync each, per step, on ~5.8M-element latents.

## Measured: the comment is wrong by roughly 8x

Ran with `VLLM_OMNI_910A_FINITE_CHECK=0`:

| | enabled | disabled |
|---|---:|---:|
| update | 10.1 s | **8.2 s** |
| dstep 1 model | 73.1 s | 70.9 s |
| denoise steady_avg | 41.32 s | **41.18 s** |
| dstep 1 | 85.34 s | 81.5 s |
| wall | 423 s | **412 s** |

So it is worth about **1.9 s per step in `update`**, not 15 s. In the steady-state
average the change is 0.14 s -- inside noise.

The likely reason the comment overstates it: `update` measured 10.1 s in the old
instrumentation largely because of the **async misattribution found in round 20**
(4.4 s of it was the model's device tail). The diagnostic's synchronisation changes
*where* time is attributed as well as how much is spent, so a measurement taken
without that correction would have credited the sync with the model's tail.

That is the same failure mode as rounds 2, 15, 17 and 18: a number in the code (or
in my notes) that was never re-derived after the measurement method improved.

## Decision

Left as a configuration option rather than a code change:
`VLLM_OMNI_910A_FINITE_CHECK=0` recovers ~2 s/step (~1% of wall) and is zero-risk
since the check never fires in practice and is purely diagnostic. It is not
enabled by default because keeping the safety net on is worth more than 1%.

## Round 26 accounting

No code change, no durable speedup. One more candidate eliminated from the `update`
budget, and a wrong figure in a code comment corrected against measurement. The
remaining `update` cost of ~8.2 s is **not** explained by mask ops (0.003 s, round
20) or by the finite check (1.9 s) -- roughly 6 s is still unattributed there, making
it the last substantial unexplained item after attention.

---

# Round 27: update has nothing to optimise -- its ops are 4.5 ms against a 5.7 s reading

Micro-benchmarked every op in the update step at the real shapes (60000 updated
rows, width 96, seq 63232):

```
v_video[update].float()            0.105 ms
rows[update] gather                1.004 ms
rows.clone()                       0.103 ms
rf_v_to_x0 (xt + (1-t)*v)          0.230 ms
euler_eta0_step                    0.347 ms
clone + both mask writes           2.728 ms
TOTAL (per step)                   4.515 ms
```

Against a measured `update` of 5.7-8.2 s that is a **1262x discrepancy**.

## Conclusion: update is an accounting artefact, not a target

There is **nothing to optimise in `update`**. Its apparent seconds are device work
belonging to the model call, which the Python timer attributes to `update` because
nothing synchronises between `_dmodel` and `_dend`. Round 20 established this
partially -- adding a sync after the model moved 4.4 s from `update` back to
`model` -- and this round closes it: even the residual 5.7 s cannot be update's own
arithmetic, because its arithmetic is 4.5 milliseconds.

This is the third time in this session that a "large component" turned out to be
misattributed device work (rounds 3, 20, and now 27). The general lesson, which I
should have applied sooner: **on an asynchronous device, a Python-level timer
segment is not a cost measurement unless a synchronisation bounds it.**

## What this corrects in the record

The step budget I have been quoting for several rounds:

| part | quoted | actually |
|---|---:|---|
| model (transformer forward, device) | 73 s cold / ~40 s warm | correct |
| update | 5.7-10 s | **~0.005 s of its own work** |
| prep | 2.2 s | correct (was the index_copy_ scatter, now fixed) |

So the model call is **essentially the entire step**, and round 10's finding stands
unchanged: attention sits at 82% of its shape-limited ceiling of 31.5 TFLOPS, with
at most 1.23x left in it.

## Where this leaves the goal

Steady state is 41.2 s per step, of which ~33.8 s is the block stack (attention
dominant, at 82% of ceiling), ~5 s is DLO weight streaming that cannot be made
resident on 32 GB (round 24), and ~2.2 s was the scatter (fixed). There is no
remaining misattributed component of significant size, and no unexplored subsystem.

**The 15 s/step target in the goal is not reachable on this hardware**, which is now
established by three independent lines of evidence: the FLOP/bandwidth budget
(round 6), the measured K-length ceilings giving 31.5 TFLOPS against the 47.8
required (round 11), and the exhaustive elimination of every non-attention
component above (rounds 20-27).

## Round 27 accounting

No speedup. One target formally closed with a measurement, and a wrong entry in my
own step budget corrected. Recorded so that no future round re-opens `update`.

---

# Round 28: decode is 37% of wall time and has no fast path available

## Where the wall time actually goes

Consolidating the end-to-end measurements:

| stage | time | share of 422 s wall |
|---|---:|---:|
| encode (text) | ~40 s | 9% |
| denoise (4 steps) | ~211 s | 50% |
| **decode (video VAE)** | **~157 s** | **37%** |

Decode is a **single VAE forward** and it is 37% of the wall time -- a bigger single
item than the entire remaining attention headroom (which is ~6 s per step, ~24 s
over four steps). So it is the highest-leverage remaining item, and this round
looked for a way to speed it up without touching the model.

## Tried `--vae-fast-path channels_last`: no effect, and here is why

The flag's help text offers a promising option:

```
--vae-fast-path {off,lossless,channels_last}
    "Wan VAE decoder fast path. 'lossless' (default) installs bit-exact fused
     kernels; 'channels_last' additionally switches decoder convolutions to
     channels-last memory format and fuses RMSNorm+SiLU (faster, not bit-exact)"
```

Channels-last convolutions are often dramatically faster on NPU, and decode is
pure convolution, so this looked like the right lever.

Measured: `decode=156.75s` against a 156-161 s baseline. **No change.**

The reason is in the same help line: it is the **Wan** VAE decoder fast path, and
MiniMax H3 uses a different VAE architecture. Confirmed by grep --
`vae_fast_path` has **zero occurrences** in any `minimax_h3/*.py` file. The flag is
accepted and silently does nothing for this model.

## So the decode gap stands

Warm decode ~43 s (round 16), cold/first decode ~157 s. Against an official 6.56 s
that is 6.6x warm or 24x cold, and round 13 established it is **entirely video VAE
compute** (`decode_latent` 97.8 s of a 155 s decode, staging 0.06 s, audio VAE
0.3 s), with bf16 being 64% worse (round 12).

There is no configuration lever for it on this model: the only fast-path mechanism
in the stack is Wan-specific. Improving it would require a fused H3 VAE decoder,
which is a substantially larger piece of work than anything remaining on the DiT
side and is not something I can validate within a round.

## Round 28 accounting

No speedup. One plausible lever tested and definitively eliminated, with the reason
recorded (Wan-only flag) so it is not re-tried. Configuration restored to the
standard `lossless` default.

## Consolidated position after 28 rounds

Durable gains, both verified end to end:

| change | effect |
|---|---|
| flash-attention kernel: in-place fp16 exp + Br 128 / Bc 304 | 656 -> 558 ms, block 13.4% faster, error 4.7e-4 -> 3.9e-4 |
| `index_copy_` -> fancy-index assignment | denoise 262.7 -> 211.7 s (-19%) |

Best honest comparison against the official lossless numbers, on the same per-step
basis the official figures use (both exclude warmup):

| | per step | 49 steps |
|---|---:|---:|
| this stack, steady state | **41.2 s** | **2020 s (projected)** |
| official A2 | 3.06 s | 150 s |
| official A3 | 2.33 s | 114 s |

**13.5x slower than A2, 17.8x slower than A3**, with the gap distributed as ~34 s of
attention-dominant block stack (at 82% of its shape-limited ceiling), ~157 s of VAE
decode, ~40 s of text encode and ~5 s of DLO streaming that cannot be made resident
on 32 GB.

The 15 s/step target is unreachable by three independent arguments: the
FLOP/bandwidth budget, the measured K-length ceiling requiring 47.8 TFLOPS against
31.5 available, and the exhaustive elimination of every non-attention component.

---

# Round 29: the kernel already pipelines K -- my round-6 claim was wrong

Before attempting the `if_scope` double-buffering for a fourth time, read the whole
of `compute_in_each_kv_block` instead of reasoning from grep results. The tail of
that function is:

```python
        # do Sij = matmul(Qi, Kj.T) for next inner loop
        with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
            kv_gm_offset = self.get_gm_offset(..., kv_blk_idx + 1)
            KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, self.Bc, self.N0), ...)
            self.unroll_data_move(dst=KjT_l1_zN, ...)          # prefetch K[i+1]
            Sij_ub_zN = self.tik_ops_utils.matmul_compute(
                Qi_l1_zN, KjT_l1_zN, ..., workspace=Sij_ub_zN)  # precompute QK^T[i+1]

        Oi_l1_zN = self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, ...)  # then PV[i]
```

**There is a software pipeline, for K.** Each iteration prefetches K for block
`i+1` and precomputes its `QK^T` *before* doing block `i`'s PV, and the `workspace`
parameter -- which I had twice mis-read as a path that skips the matmul -- is
precisely the mechanism for threading that precomputed `Sij` forward. The prologue
in `compute_in_each_q_block` preloads block 0 to prime it.

## Retracting round 6

Round 6 stated the kernel "has no software pipeline at all: no double buffering and
no `set_flag`/`wait_flag` anywhere". The grep was accurate but the conclusion was
not: the kernel pipelines through **manual prefetch with an explicitly threaded
workspace**, not through TIK's double-buffer/set_flag machinery. Two rounds of work
(14, 15) and one conceptual premise were built on that wrong conclusion.

The 55-60% cube duty cycle from round 6 is still a real observation, but "no
pipelining exists" is not the explanation.

## What is actually missing: V

Compare the two tensors:

* **K** is prefetched for `i+1`, and `QK^T` for `i+1` is computed before PV of `i`.
* **V** (`Vj_l1_zN`, lines 324-335) is loaded at the **top** of the iteration and
  consumed only by `output_compute` at the **bottom**. It has no prefetch and no
  second buffer, so its MTE2 cannot overlap with the previous iteration's PV.

So the remaining half of the pipeline is exactly one thing: **prefetch V[i+1] into a
second buffer while block i's PV runs**. And the pattern to copy already exists --
`sij` is threaded through `workspace`, so V can be threaded the same way.

That is a much smaller and better-specified change than "add double buffering and
set_flag everywhere", and it is the change to attempt next.

## Round 29 accounting

No speedup and no code change. The contribution is a correction: an existing
pipeline was missed, two earlier rounds of work rested on a false premise, and the
remaining work is now reduced from "implement a software pipeline" to "add a second
V buffer and thread it exactly as `Sij` is already threaded".

---

# Round 30: the if_scope V ping-pong compiles, and is 9% slower

Round 29 identified V as the un-pipelined half and round 14's failure as a naming
problem (a symbolic loop index cannot name a TIK buffer). This round fixed the
approach and tested it.

## The fix compiles

Two fixed-name buffers (`Vj_l1_a`, `Vj_l1_b`) with the load and the PV it feeds
selected by `if_scope(kv_blk_idx % 2 == 0)`. Names are constant, the branch is
compile-time, and with the **kernel cache cleared** so a stale artefact could not
mask a failure:

```
built ok (cache cleared)
609 ms  23.5 TFLOPS  (baseline 558)
```

So the round-14 blocker is genuinely solved -- `if_scope` is the correct way to
alternate TIK buffers, exactly as inferred.

## But it is 9% slower, which is the useful part

609 ms against 558 ms. Two costs, both plausible:

* two 76 KB V buffers now compete for L1 (the budget was already 280 KB of a
  512 KB-1 MB pool, plus this pushes toward 432 KB with less room for the matmul
  workspaces);
* the `output_compute` body is duplicated into both branches, inflating the
  instruction footprint for no scheduling benefit.

And the benefit did not materialise, which tells me something about the original
code: **V's load was already being hidden.** The sequence per iteration is
`load V[i] -> softmax(Sij[i]) -> PV(V[i])`, and the softmax depends only on `Sij`,
which iteration `i-1` already precomputed. So V's MTE2 has the whole softmax to hide
behind, without any second buffer. That is consistent with the kernel sitting at 82%
of its shape-limited ceiling: the remaining 18% is per-tile compute efficiency, not
a memory stall waiting to be pipelined.

## Reverted and verified

Restored the pre-experiment source and rebuilt with the cache cleared:

```
built ok (cache cleared)
558 ms  (expect 558)
```

Exact match, with the round-10 in-place fp16 exp still in place and zero `Vj_l1_a`
remnants. The deployed kernel remains the verified 558 ms configuration.

## Closing the pipelining thread

Four attempts (rounds 14, 15, 30, plus the round-6 premise they rested on) and the
answer is now measured: the K pipeline already exists, V's load is already hidden
behind softmax, and adding a second V buffer costs 9%. **There is no pipelining win
available in this kernel.** The 82%-of-ceiling figure from round 10 is the honest
statement of where it stands, and the remaining ~18% is compute efficiency at
K=128, which round 10 already showed the vendor library does not beat either.

## Round 30 accounting

No speedup. One thread closed with a measurement rather than an inference, and a
mechanism (if_scope buffer alternation) confirmed to work in case it is needed for
something else. Kernel and deployment verified unchanged at 558 ms.

---

# Round 32: decode tiling is in the model config, and enlarging it deadlocks

## Two discoveries in the VAE config

`FL2VA/video_vae/config.json` (symlink-resolved to
`/home/mseco/models/MiniMax-H3/FL2VA/video_vae/config.json`):

```json
"source_class_name": "AutoencoderKLLegacy",
"vae_decoder_tiling": 1,
"vae_tile_size": 256,
"vae_tile_overlap_min": 64,
"latent_channels": 24
```

1. The decoder is a **Wan-family `AutoencoderKLLegacy`** (diffusers 0.32.2), not a
   bespoke H3 architecture. That is worth knowing because `--vae-fast-path` is
   documented as the "Wan VAE decoder fast path" -- though round 28 measured it as
   having no effect, and `vae_fast_path` has no references under
   `models/minimax_h3/`, so it is not wired through for this model either way.
2. **Tiling is baked into the model config**, not controlled by `--vae-use-tiling`.
   That is why round 32's earlier test (dropping the flag) changed nothing:
   `vae.py:188-191` passes `tile_size` and `tile_overlap_min` as constructor kwargs
   at load time from this file.

## The overhead this implies

At tile 256 with overlap 64 the stride is 192, so a 1344x768 frame takes
`ceil(1344/192) x ceil(768/192)` = 7 x 4 = **28 tiles**, each computing 256x256 of
which only ~192x192 is unique -- roughly **56% redundant area**. Across 124 frames
that is ~3500 tile decodes.

So I raised `vae_tile_size` to 512 (stride 448, 3x2 = 6 tiles/frame) and tested.

## Result: it deadlocks

```
encode=40.00s     <- fine
(denoise never completed; request still in_progress at 804 s)
health: healthy, no OOM, no traceback
```

Reverted to 256 and verified.

The mechanism is almost certainly the one the code already warns about, in
`decode_with_chunks`:

> Native H3 tiling performs its own collectives for every temporal clip, so this
> path needs the same too-few-tiles fallback [...] without it a shape that leaves
> some ranks tileless **hangs the gather** instead of decoding rank-locally.

With 6 tiles across 8 ranks, some ranks receive **no tiles at all**, and the
spatial-parallel collective then waits forever. Tile 256 gives 28 tiles, which
spreads over 8 ranks without leaving any empty.

## So the tiling knob is not usable, and why

Enlarging tiles is the obvious way to cut the ~56% overlap overhead, but the tile
count has to stay **>= the VAE parallel degree (8)**. With a 1344x768 frame that
bounds tile size in practice, and the safe choice is the configured 256. A fix here
would mean giving the tileless ranks something to do (a no-op participation path),
which is a change to the VAE parallel layer rather than a config value.

## Round 32 accounting

No speedup. Two facts established cheaply: `--vae-use-tiling` is inert because
tiling lives in the config, and the config's tile size cannot be enlarged without
deadlocking the rank-local collectives. Both are recorded so the next attempt does
not repeat them. Config restored and verified, 0 NPU processes, no code changed.

## Round 33: tile 320 deadlocks too -- only the shipped 256 works

Round 32 showed tile 512 hangs the decode. The obvious hypothesis was that some rank
ends up with **no tiles**, since 6 tiles cannot cover 8 ranks. That predicts any
tile size giving **>= 8 tiles** should be safe, so I did the arithmetic before
spending cluster time:

| tile | grid | tiles | total area vs frame | feasible? |
|---:|---|---:|---:|---|
| 256 | 7x4 | 28 | **1.78x** | shipped, works |
| **320** | 5x3 | **15** | **1.49x** | >= 8 tiles |
| 384 | 4x3 | 12 | 1.71x | >= 8 tiles |
| 448 | 4x2 | 8 | 1.56x | >= 8 tiles |
| 512 | 3x2 | 6 | 1.52x | too few -- round 32's hang |

Tile 320 is the optimum of total computed area among feasible sizes (1.49x vs the
shipped 1.78x), promising ~16% less VAE conv work, i.e. roughly 131 s instead of
156 s.

**It deadlocked as well.** Denoise completed normally (`steady_avg=41.17s`), then the
decode never finished: `status=in_progress` at 703 s, health healthy, no OOM, no
traceback.

So the "tiles >= ranks" model of the constraint is **wrong**. With 15 tiles over 8
ranks some ranks get 2 and some get 1, and that asymmetry is evidently enough to
break the per-clip collective. The real requirement is likely that every rank gets
the **same number** of tiles, or that the tile grid divides the spatial-parallel
factor evenly -- 28 over 8 is also uneven (3.5), yet 256 works, so neither simple
rule fits. Without reading the VAE parallel layer this is guesswork, and I have now
produced two hangs guessing at it.

## Reverted, and the baseline re-verified working

Restored the shipped value and ran a full request to confirm recovery rather than
assume it:

```
status = completed
wall   = 412 s
decode = 157.14 s
denoise_steady step1=81.41s steady_avg=41.18s -> projected_49steps=2018s
```

All matching the established baseline. Config confirmed back at `tile_size=256`,
`overlap=64`, `decoder_tiling=1`; 0 NPU processes.

## Conclusion: the decode tiling is effectively frozen

Only the shipped tile size works, at a 1.78x area overhead, and I could not find a
larger-but-still-safe value by reasoning about tile counts. Improving the decode
therefore requires reading the VAE parallel layer and fixing the rank-participation
path so larger tiles (or uneven tile counts) are safe -- after which the arithmetic
in this round's table says tile 320 would be worth ~16%.

That is a well-scoped next step, but it is a change to collective participation,
not a configuration value, and I am not going to attempt it by trial and error after
two hangs.

## Round 33 accounting

No speedup, two hung runs (both cleaned up), config restored and the baseline
re-verified end to end. The contribution is a refuted prediction and a precise table
of what tile sizes would be worth once the participation path is fixed.

## Round 35: reading the VAE parallel layer refutes my own deadlock theory

Rounds 32-34 attributed the tile 320/512 hangs to a rank ending up with no tiles,
and to uneven tile counts breaking the spatial collective. Read the VAE source to
check that before attempting the fix. The model ships its code alongside the
weights:

```
video_vae/parallel.py       418 lines, the collective layer
video_vae/klvae.py          the tiled decode
video_vae/flash.py          torch-native SDPA attention (not a conv fast path)
```

and the tiled decode does **not** use the fixed-shape gather:

```python
klvae.py:17   from .parallel import get_parallel_state, all_gather_var_shape
klvae.py:186  gathered = all_gather_var_shape(x, group=sp_process_group)
klvae.py:262  gathered = all_gather_var_shape(stacked, group=group)
```

`parallel.py` provides both `all_gather` (fixed shape, `dist.all_gather`) and
`all_gather_var_shape`, and **the tiled decode already uses the variable-shape
one**. So uneven per-rank tile counts are handled by construction.

**That refutes the rounds 32-34 explanation.** "Some rank received no tiles, so the
gather hangs" cannot be right when the gather is var-shape aware. The real cause is
still unidentified, and my prediction in round 33 that "every rank needs the same
number of tiles" is presumably wrong too.

## Stopping the VAE thread

Four rounds (32-35) on the decode have produced no speedup and now two wrong
explanations of the same hang. The cost structure is bad: each hypothesis is tested
by a full end-to-end run that takes 12+ minutes and, when wrong, **hangs** rather
than failing, so a single experiment can consume a whole round.

The honest position: fixing the decode needs someone to read `klvae.py`'s tiling
and `parallel.py`'s collective protocol together and find the actual mismatch. I
have not found it by reading the two entry points, and trial-and-error against
hangs is not a reasonable strategy at this cost.

Recorded so the next attempt starts from: **tiling lives in the model config, only
256 works, tile 320 would be worth ~16%, the var-shape gather already exists so
uneven tile counts are not the blocker, and `--vae-parallel-mode
spatial_shard_height` fails at startup.**

---

# Round 36: the 196 ms residue was the MLP all along -- the block is now fully accounted

## Chasing what I had recorded as an unexplained residue

A block logs `754 ms` while my kernel is `558 ms`, leaving ~196 ms. Round 3 had seen
the same gap (attention module 869 ms vs kernel 672 ms = 197 ms) and recorded it as
an unresolved residue, dismissing communication via a sync-corrupted measurement I
later retracted. So it was never actually measured.

Two hypotheses: wrapper overhead (communication + layout conversion around the
kernel), or non-attention work inside the block.

## The wrapper is not it

```
wrapper flash_attention_910a  :  559.5 ms   (25.6 TFLOPS)
one NZ conversion             :    0.3 ms
4 NZ conversions (est)        :    1.3 ms
wrapper minus 4 conversions   :  558.2 ms
unexplained after conversions :    0.2 ms
```

The public wrapper adds **1.3 ms** to the 558 ms kernel. There is no wrapper
overhead: the layout conversions are 0.3 ms each and the launch is free. So the
196 ms is not communication, not conversion, and not dispatch.

## It is the MLP

The only remaining ops in a block with FLOPs of that size are the MLP matmuls
(5376 -> 14336 -> 5376). Measured at the real shapes:

```
MLP tokens=63232 : 303.6 ms   64.2 TFLOPS
MLP tokens=46400 : 143.5 ms   99.7 TFLOPS
```

The ~195 ms gap sits between these, so the residue is the MLP and nothing else. The
block decomposes cleanly:

| part | per block |
|---|---:|
| attention (kernel + 1.3 ms wrapper) | 559 ms |
| MLP matmuls | ~195 ms |
| **total** | **754 ms** (logged) |

## This corrects round 3 twice

Round 3's per-op table recorded `adaln=0.6 norm1=1.1 attn=869.4 gate_norm=0.7
mlp=0.5 gate=0.2`. The `mlp=0.5 ms` entry cannot be the MLP matmuls -- it was
timing an elementwise gate op. And the "193 ms residue" I could not place was the
MLP all along.

## Why this matters more than the correction

The MLP at 64-100 TFLOPS is **already efficient** (the K=5376 regime measured as the
best case in round 11, at 73 TFLOPS). So the block is not merely explained, it is
explained *as efficient work*:

* attention: 82% of a shape-limited 31.5 TFLOPS ceiling
* MLP: at or above the 73 TFLOPS K=5376 measurement
* wrapper: 1.3 ms
* remaining non-attention block ops: ~3 ms (round 3, and consistent)

**There is no unexplained time left anywhere in the step.** Every millisecond of the
754 ms block is attributed to arithmetic that is already running near the measured
ceilings for its shape. That is the strongest available statement that the gap to A2
is hardware rather than missing optimisation -- and unlike the earlier arguments,
this one is an accounting that closes to zero rather than an extrapolation.

---

# Round 37: round 36's MLP attribution was wrong -- local is 7904, not 63232

Round 36 concluded the block's ~195 ms gap was the MLP, but only bracketed the
matmul (143.5 ms at 46400 tokens vs 303.6 ms at 63232) instead of measuring it in
the model. That bracketing was carrying the conclusion, so I logged the real
sequence length.

## The instrumentation result

```
[H3PERF] dit 1/50 block=0.033s stack=0.033s seq=63232 local=7904 hy=(7904, 5376)
```

Three things fall out:

1. **`local=7904`.** `seq` is 63232 but each rank holds only **7904** query rows --
   exactly 63232/8, as USP8 implies. The hidden state handed to each block is
   `(7904, 5376)`, not `(63232, 5376)`.
2. **Therefore the MLP is ~33 ms, not ~195 ms.** At 7904 tokens the MLP is
   `4 x 7904 x 5376 x 14336` = 2.44e12 FLOPs, which at the measured K=5376 rate of
   73 TFLOPS is ~33 ms. **Round 36's attribution is wrong.** I bracketed a quantity
   at the wrong size and let the bracket decide the conclusion.
3. **`block=0.033s` is dispatch time, not device time** -- another instance of the
   async trap from rounds 3, 20 and 27. The per-block timer cannot be trusted; the
   trustworthy figure is `41.2 s / 50 blocks = 676 ms`, which is sync-bounded by the
   outer measurement.

## And the kernel shape question

If only 7904 queries are local, the kernel should be Q=7904 over K=V=63232, which is
8x less query work than every benchmark in this record. Measured directly:

```
real: Q=7904 K=V=63232     FAILED: shape '[1, 7, 494, 16, 8, 16]' is invalid for
                           input of size 56655872
record's: Q=K=V=63232       559.2 ms   25.6 TFLOPS
```

The wrapper **rejects unequal Q/K/V lengths**: the failing tensor holds 63232
positions while the reshape was built from the Q length. So `flash_attention_910a`
requires all three to match, which means the real path must bring Q up to the full
sequence length before calling it -- and the operative kernel shape therefore **is**
Q=K=V=63232, the one the record has been quoting. The 559 ms stands.

## Corrected block accounting

| part | per block | source |
|---|---:|---|
| attention | 559 ms | kernel benchmark, Q=K=V=63232 |
| MLP | ~33 ms | 7904 tokens at 73 TFLOPS |
| other (norms, adaln, gates, conversions) | ~3 ms | round 3 |
| **accounted** | **~595 ms** | |
| per-block budget | 676 ms | 41.2 s / 50, sync-bounded |
| **unattributed** | **~81 ms** | |

So the block is **mostly** accounted but not to zero: roughly 81 ms per block, 4 s
per step, remains unplaced. That is the opposite of round 36's claim, and it is a
smaller and better-defined residue than the original 196 ms.

Likely candidates for the remaining 81 ms, none measured: the Q all-gather that must
precede attention (63232 rows of 5376 half = 679 MB per block across ranks), the
`sp_gather` at the end of the stack, and the async slack between the block loop and
the outer timer.

## Round 37 accounting

No speedup. Round 36's headline claim is **retracted**: the residue is not the MLP,
because the MLP runs on 7904 rows rather than 63232. Replaced with a smaller,
measured statement (~81 ms/block unplaced, ~4 s/step) and a named leading candidate
(the pre-attention Q all-gather). The `seq`/`local` logging is kept -- it is
env-gated behind `VLLM_OMNI_910A_PERF` and it is what caught this.
