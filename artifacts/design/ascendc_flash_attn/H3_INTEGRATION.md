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
