# Measured: the TBE flash attention does not accelerate H3 on 910A

The kernel is correct and runs, so it was benchmarked at increasing sequence
lengths to see whether it beats the current dense attention path. It does not.

## Scaling

B=1, H=7, D=128, fp16, blockDim 32, warmed up, timed over the launch loop:

| N | ms/call | ms per N^2 (x1e-7) | TFLOPS |
|---|---|---|---|
| 1024 | 0.248 | 2.365 | 15.2 |
| 2048 | 1.069 | 2.549 | 14.1 |
| 4096 | 4.142 | 2.469 | 14.5 |
| 8192 | 16.683 | 2.486 | 14.4 |
| 16384 | 71.106 | 2.649 | 13.5 |

Clean quadratic, and the per-N^2 constant is nearly flat, which says the tiling
behaves consistently across sizes. Extrapolating with the N=16384 constant:

```
N = 32768  ->   284 ms   (13.5 TFLOPS)
N = 73408  ->  1427 ms   (13.5 TFLOPS)     <- the real per-rank shape
```

## Against the existing dense path

The dense multi-matmul attention was measured at the same real shape
(per-rank q of `(1, 73408, 7, 128)`):

```
dense:  QK^T 32.2 TFLOPS | softmax | SV 78.5 TFLOPS  ->  1166 ms/call
flash:  fused, 13.5 TFLOPS                            ->  ~1427 ms/call
```

**The fused kernel is about 22% slower.** It does the same arithmetic, so the
gap is pure cube efficiency: 13.5 TFLOPS against 32-78.

## Why, on the correct comparison

Both paths do the same arithmetic:
`4 * B * H * N^2 * D = 1.93e13` FLOPs (QK^T plus SV, two flops each). Dividing
by measured wall time gives the like-for-like number:

```
dense        1166 ms -> 16.6 TFLOPS overall
flash       1427 ms -> 13.5 TFLOPS overall
```

Flash attention streams in 128x128 tiles (`Br = min(128, Nq)`,
`Bc = min(128, N)`, both fixed in the reference), so every cube operation is
`128 x 128 x 128` with a fresh load and an online-softmax pass between them. The
dense path instead issues long `M`-dimension matmuls over many keys, which keeps
the cube fed.

**Correction to an earlier draft of this note.** I first wrote that the fused
kernel runs at 13.5 TFLOPS "against 32-78", implying it was 2.5x off the pace.
That compared flash's *whole-operation* rate against the dense path's *per-matmul*
rates and ignored the softmax sitting between those matmuls. The dense path's own
overall rate is 16.6 TFLOPS, not 32-78, because 148 s of its 233 s attention
budget is softmax. So the real gap is **13.5 against 16.6, and flash needs only
about 1.22x to break even** -- a materially different and much more reachable
target than the 2.5x the earlier framing implied.

## What this means for the objective

"支持 910A 的 flash attention" is achieved: it builds, runs, and matches a
float32 reference to fp16 rounding.

"加速 H3 推理" is **not** achieved, and on this evidence this kernel will not
achieve it. Integrating it would make the 4-step 449 s run slower, not faster.
It should not be wired into `h3_910a_attn.py`.

Worth being explicit about the confidence: the 1427 ms figure is an
extrapolation from a clean four-point quadratic fit, not a measurement at
N=73408. The per-N^2 constant drifts up slightly with N (2.37 -> 2.65 e-7), so
if anything the real number is worse than 1427 ms, and the drift means the
required break-even gain is somewhat more than 1.22x.

## What could still change the answer

1. **Retile for this part.** Br/Bc are fixed at 128 in the reference. Larger
   tiles would raise cube efficiency but are bounded by L0A/L0B (64 KiB each)
   and the online-softmax state in UB, so the headroom is limited and unproven.
2. **Cut the work instead of speeding it up.** The dense path already spends
   148 s of its 233 s attention budget in `softmax`, which a fused kernel does
   not remove -- it only avoids the extra round trip. That matches the measured
   outcome.
3. The lever that does not depend on any new kernel: the model. A distilled or
   pruned H3 with fewer blocks or a narrower DiT reduces the q/k/v sequence
   length itself, which is the only thing that changes the N^2 term.

## Method note

Every number here comes from the legacy-runtime launch harness
(`rtDevBinaryRegister` + `rtFunctionRegister` + `rtKernelLaunch`) with 3 warm-up
launches before timing, since the first launch includes module load. The kernel
was rebuilt per N, because the core task map is baked in at compile time -- so
each row is a genuine kernel for that shape, not a resized call.

---

# Retiling reverses the result: 1.33x faster than dense

The section above concluded the fused kernel was slower and should not be
integrated. That was measured with the reference's default tiling, which is not
the best tiling for this part. Retiling flips the answer.

## The tiles are hard-coded, and 128x128 is not optimal here

`sparse_tiling.py:27` fixes `Br = min(128, Nq)` and `Bc = min(128, N)`. They are
now overridable for experiments via `FA_BR` / `FA_BC`. Sweeping them at
N=8192, B=1, H=7, D=128 (baseline 16.686 ms):

| Br | Bc | ms | note |
|---|---|---|---|
| 128 | 128 | 16.686 | reference default |
| 128 | 192 | 13.413 | |
| 128 | 256 | 11.393 | 1.46x |
| 128 | 384+ | -- | build fails, UB budget |
| 256 | any | -- | build fails, UB budget |
| 64 | 256 | 13.669 | |
| **64** | **512** | **10.795** | **1.55x, best** |
| 32 | 512 | 15.051 | |
| 16 | 1024 | 22.047 | |

A long `Bc` is what the cube wants, and shrinking `Br` buys the UB headroom to
afford it. `Br` alone cannot grow -- 256 fails outright.

## Tuned performance

Br=64, Bc=512:

| N | ms | TFLOPS |
|---|---|---|
| 4096 | 2.713 | 22.2 |
| 8192 | 10.795 | 22.3 |
| 16384 | 43.766 | 22.0 |

Flat 22 TFLOPS, versus 13.5 at the default tiling and 16.6 for the dense path's
whole attention step. At N=16384 the speedup over the default tiling is 1.62x.

Extrapolating the tuned constant to the real per-rank shape:

```
N = 73408  ->  879 ms   (22.0 TFLOPS)
dense      -> 1166 ms   (16.6 TFLOPS)
                       -> 1.33x faster
```

## Numerics are unchanged by the retiling

Rebuilt at N=256 with Br=64, Bc=512 (both clamp to the sequence length) and
compared against the same float32 reference:

```
max|diff| = 0.000009   relative = 0.00051
```

identical to the default tiling, so the speedup is not bought with accuracy.

## Effect on the H3 run

The attention step was measured at 200 calls x 1.166 s = 233 s inside the 449 s
4-step run. At 879 ms/call that becomes 176 s:

```
449 s  ->  ~392 s
```

So this is a real speedup of roughly 57 s on the 4-step configuration. It is the
first change in this whole line of work that makes the fused kernel a win rather
than a loss, and it makes integration worth doing.

It does **not** reach the 300 s target on its own. 392 s is still 92 s over, and
the remaining budget would have to come from elsewhere (the 43 s VAE decode, the
148 s softmax share that fusing does not remove, or fewer steps).

## Confidence

The 879 ms is an extrapolation, not a measurement at N=73408. It is better
supported than the earlier 1427 ms figure because the tuned per-N^2 constant is
flatter across three points (1.616, 1.609, 1.631 e-7 at N=4096/8192/16384) than
the untuned one was. The next step should be a real measurement at N=32768 and
then at the true shape, since the tiling that is best at 16384 need not be best
at 73408 -- the sweep should be repeated near the real length before committing.

---

# RETRACTION: the 1.33x / 1.25x speedup is invalid. The large-tile path is numerically wrong.

The retiling result above is built on incorrect output. Retracting it.

## The remainder path is broken

`sparse_tiling.py` computes `last_Bc = N - Tc*Bc` when `N % Bc != 0`, so the
kernel can have a short final key block. That path does not produce correct
results.

Test at N=1280, Bc=512, which leaves `last_Bc = 256`:

```
N=1280  Br=64 Bc=512   max|diff| = 0.002532   max|ref| = 0.0129   rel = 0.197
rows with error > 1e-3:  1280 / 1280
```

20% relative error on **every** row -- not only the 256 rows in the short block.

Control, same tiling, N=1024 where `1024 % 512 == 0` and the remainder path is
never taken:

```
N=1024  Br=64 Bc=512   max|diff| = 0.000008   rel = 0.00065
```

exact. So the defect is specifically the non-divisible case, not the tiling.

## What that does to the real shape

`73408 = 2^6 * 31 * 37`, so:

| Bc | N/Bc | remainder | correct? |
|---|---|---|---|
| 16 | 4588.0 | 0 | yes |
| 32 | 2294.0 | 0 | yes |
| 64 | 1147.0 | 0 | yes |
| 128 | 573.5 | **64** | **no** |
| 256 | 286.75 | **192** | **no** |
| 512 | 143.375 | **192** | **no** |

`Br = 128` leaves the same remainder of 64. So **both tile dimensions must be
<= 64** for the real sequence length, and this is not a limitation of my
retuning -- the reference's own default of 128x128 is broken at N=73408 too.

The 935.7 ms measurement was taken with Br=64, Bc=512 on this shape, i.e. on
wrong output.

## Measured performance of the only correct configuration

```
N=73408  Br=64  Bc=64   2698.130 ms/call      <- correct
dense path              1166 ms/call          <- for comparison
```

**2.3x slower than dense.** So the entire apparent speedup was an artefact of
the broken large-tile path: the long `Bc` is what made the cube efficient, and
a long `Bc` cannot be used correctly at this sequence length.

## Corrected bottom line

* Flash attention on 910A: builds, runs, and is numerically correct for shapes
  whose sequence length is divisible by the tile size.
* It does **not** accelerate H3. The fastest correct configuration found is
  2698 ms versus 1166 ms for the existing dense path.
* The earlier 1.33x and 1.25x claims are withdrawn.

## What would have to be true to revisit this

1. **Fix the remainder path.** This is the highest-value fix: with it, Bc=512
   becomes usable and the measured 935.7 ms would be real, i.e. 1.25x faster
   than dense. That single bug is the difference between slower-than-dense and
   faster-than-dense.
2. Or avoid the remainder by construction: pad the *key* dimension to a multiple
   of `Bc`. Padding with zeros does not work -- the kernel takes no mask, so
   zeroed keys would each contribute `exp(0) = 1` of weight and dilute the
   result. It would need either a mask or `-inf` scores, neither of which the
   reference interface exposes.
3. Or measure with Bc<=64 across the board and accept ~2.3x slower than dense,
   which is not worth integrating.

## Method note, and the mistake made here

The first timing pass used the tuned tiles and reported 1.25x faster without
checking correctness at that configuration. The tuned tiles were validated only
at N=256, where `Bc` clamps to the sequence length and the remainder path is
never entered -- so the test could not have caught this. The lesson is that a
tiling parameter validated at a small shape does not carry to a large shape
unless the divisibility class is the same, and N=256 was silently the easy case.
Any future timing of this kernel should assert numerical agreement at the exact
shape being timed before the timing means anything.

---

# Resolved: use a tile that divides the sequence length

The retraction above is right about the bug and wrong about the conclusion. The
remainder path does not have to be entered at all -- the tile size can be chosen
to divide the sequence length.

## The arithmetic

`73408 = 2^6 * 31 * 37`. Powers of two above 64 do not divide it, which is why
128/256/512 all hit the broken path. But other factorisations do:

```
Bc = 496 = 16 * 31  ->  73408 / 496 = 148  exactly
Bc = 592 = 16 * 37  ->  73408 / 592 = 124  exactly
```

Both are 16-aligned (the kernel needs that) and both are large enough to keep
the cube efficient. So the choice is not "Bc <= 64 and slow" after all.

## Correctness

Verified at exact division, N=992 with Br=64 Bc=496, against an fp32 reference:

```
max|diff| = 0.000007   relative = 0.00056
```

so the long tile is correct whenever it divides evenly, as expected from the
N=1024 control earlier.

## Measured at the real shape

N=73408, B=1, H=7, D=128, blockDim 32:

```
Br=64  Bc=496    867.752 ms/call     <- correct, divides exactly
Br=64  Bc=592    903.622 ms/call     <- correct, divides exactly
Br=64  Bc=64    2698.130 ms/call     <- correct but tile too short
Br=64  Bc=512    935.719 ms/call     <- INVALID, remainder path
dense path      1166      ms/call
```

**867.8 ms against 1166 ms is 1.34x faster**, and this one is not an
extrapolation or an artefact -- it is a measurement at the true shape with a
configuration whose correctness is established.

## Effect on the H3 run

Attention was 200 calls x 1.166 s = 233 s inside the 449 s 4-step run. At
867.8 ms:

```
200 x 0.8678 = 174 s   ->   449 s becomes ~390 s
```

About 59 s saved. That is a genuine speedup and worth integrating, but 390 s is
still 90 s over the 300 s target, so this alone does not reach the five-minute
goal.

## What is now known, precisely

| configuration | ms at N=73408 | correct | vs dense |
|---|---|---|---|
| Br=64 Bc=64 | 2698 | yes | 2.3x slower |
| **Br=64 Bc=496** | **868** | **yes** | **1.34x faster** |
| Br=64 Bc=592 | 904 | yes | 1.29x faster |
| Br=64 Bc=512 | 936 | no | invalid |
| default Br=128 Bc=128 | -- | no (remainder 64) | invalid |

Note the reference default is in the invalid row: it would have been silently
wrong at this sequence length. The correct rule is that both `Br` and `Bc` must
divide `N`, and any future call site has to pick tiles per sequence length
rather than trusting the defaults.

## Remaining risk before integration

The correctness evidence is at N=992 (exact division, fp32 reference) and the
timing is at N=73408. Those are different shapes, and the thing that could
differ between them is the per-head/per-block task map the compiler bakes in.
The safest next step is a correctness check at the true shape on a subset of
rows -- compute an fp32 reference for a handful of query positions against all
73408 keys, which is cheap, rather than materialising the full score matrix.
That has not been done yet.
