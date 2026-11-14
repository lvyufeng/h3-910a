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
