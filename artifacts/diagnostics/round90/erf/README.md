# Round90 / task-36 — Float32-vector erf-GELU kernel for the four vision mergers

**SOURCE ONLY.** Nothing in this directory has been compiled, launched or imported
on an NPU by its author: no SSH, no NPU/GPU work, no model run, no CANN build, no
installation, no production edit and no commit. It contains no numerical result
and no speed result. Only the parent driver may compile and execute it.

## Why

Round89 measured, on the **real frozen activation** of the vision mergers, that the
native NPU `GELU(approximate='none')` differs from the CPU FP16 reference by
`3.10e-4` (`merger.act_fn`) and `4.33e-4` (`deepstack_merger_list.0.act_fn`)
relative RMS, while the 27 block MLPs use `tanh` and are already within `5e-6`.
Two further Round89 facts pin the cause on the operator, not the call site:

* CPU `F.gelu` in Float32 then one Half cast is within `2.89e-7` / `1.13e-6` of CPU
  `F.gelu` directly in Half, and
* handing the native NPU operator a Float32 operand and casting afterwards is
  **bit-identical** to the native Half output.

So the native NPU erf-GELU arithmetic itself is coarse. This kernel computes the
original erf semantics explicitly in Float32 and narrows exactly once.

## What it computes

```
x (Half GM) -> x (Float32 UB)                       h_cast, the only widening
            -> z = |x| / sqrt(2)                    vec_abs + vec_muls
            -> t = 1 / (1 + p*z)                    vec_rec + 2 Newton steps
            -> poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
            -> erf_abs = 1 - poly*exp(-z*z)         A&S 7.1.26, vec_exp
            -> gelu = 0.5*x + 0.5*|x|*erf_abs       vec_mul/vec_add
            -> ONE Float32 -> Half cast -> GM
```

Coefficients are Abramowitz & Stegun 7.1.26 (`p = 0.3275911`,
`a1..a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429`,
maximum absolute error `1.5e-7` in exact arithmetic).

### Why `|x|` instead of a sign select

`erf` is odd, and `x*sign(x) == |x|`, so the original formula collapses to the
sign-free identity `gelu(x) = 0.5*x + 0.5*|x|*erf_abs(|x|/sqrt2)`. This matters
because the reviewed SDK exposes **no elementwise compare**: `vec_sel` needs a
per-element selection tensor that only a compare could produce. The identity also

* gives exactly `0` at `x == 0` (`t=1`, `poly(1)=1`, `exp(0)=1`);
* never divides by a vanishing denominator, since `1 + p*|z| >= 1`;
* makes the far negative tail exact — once `exp(-z*z)` underflows to `0`,
  `erf_abs == 1` and `0.5*(x + |x|) == 0`.

The explicit `|x|` buffer is not cosmetic: writing the last term as
`(z*erf_abs)*(1/sqrt2)` rounds the same product twice and leaves a spurious
`+2e-3` at `x == -65504`. The symmetric form removes that asymmetry, and the
stdlib tests pin it.

## Offline stability analysis (no hardware involved)

`scalar_emulation()` reproduces the kernel's op sequence step by step in IEEE
binary32 (`struct` packing) so the formula can be audited before any compile:

| Check | Result |
|---|---|
| max abs error vs double-precision erf-GELU, 6001-point sweep | `1.9e-6` |
| `x = 0` | exactly `0.0` |
| far negative tail (`-65504, -1000, -100, -30`) | exactly `0.0` |
| positive tail (`100`, `65504`) | exactly `x` |
| `erf_abs` range | `[0, 1]` |
| Newton robustness, reciprocal seed error up to `2^-8` (0.4%) | unchanged accuracy |
| predicted Half output vs Float32-exact erf-GELU, N(0,1) sample | `6.6e-6` relRMS, max 1 Half ULP |

That last row is a **prediction from the emulation**, not a measurement: it
suggests ~47× better leaf agreement than the native `3.1e-4`. Hardware `vec_exp`
ULP behaviour is not modelled, and `x == -0.0` yields `+0.0` where the CPU yields
`-0.0` (numerically equal, bit-different, recorded rather than claimed away).

## Files

| File | Role |
|---|---|
| [erf_gelu90.py](<erf_gelu90.py>) | `build_kernel(numel, out_dir)`, plan/emission tables, Float32 emulation, `ErfGeluAdapter90` |
| [test_erf_gelu90.py](<test_erf_gelu90.py>) | 41 stdlib-only tests (no torch, no TIK, no NPU) |
| [manifest90.json](<manifest90.json>) | hashes, launcher ABI, geometry, math pins, non-claims |

## Adapter scope

`ErfGeluAdapter90(vision, runtime, kernel_dir)` patches **only** the four original
`approximate='none'` merger activations — `merger.act_fn` and
`deepstack_merger_list.0/1/2.act_fn` — via one `types.MethodType` instance
`forward` each. It refuses any leaf whose mode is not `none`, and refuses to
proceed unless the vision inventory is exactly 27 `tanh` + 4 `erf` GELUs. With the
switch off it calls the captured original bound method, which is the exact native
path; `torch.nn.GELU`, `torch.nn.functional`, the biased linears and the `tanh`
block GELUs are untouched. `uninstall()` restores under a `__self__`/`__func__`
identity guard and drops the instance attribute so a later control sees a pristine
module.

## Contract facts the parent needs

* 5-GM ABI: `[Half input, Half aux1, Half aux2, Half aux3, Half output]`. The
  three aux pointers are unused by the math and exist only to keep the proven
  five-pointer launcher ABI; they are passed explicitly to `BuildCCE`.
* Real geometry `2016 x 4608 = 9289728` elements, `tile 4096` → 2268 tasks,
  30 cores, 76 steps; smoke `8192` → 2 tasks, 2 cores, 1 step.
* Per tile: 2 `data_move` bursts of 128 blocks (4096 B, the proven R83 ceiling),
  FP32 vector ops with `mask=64`, `repeat_times=64`, `rep_stride=8`.
* UB plan 147456 B, below the 196608 B safe budget. 37 instructions per tile,
  exactly 2 casts, one of them the final narrowing.
* `Ascend910B` compile SoC / short `Ascend910` runtime, 30 physical cores, no
  BF16, no A2-only feature, no assumed `COREDIM` 32.
* Tensor names are refused if they collide with a generated C identifier:
  Round58's build failed because a TIK tensor named `half` shadowed the `half`
  type. A test pins this hazard.

## Honest limits

The module is ~990 lines, above this task's "~500 if possible" preference. The
excess is functional, not boilerplate: locked/atomic build cache with resolved-path
guards, the Float32 emulation and its stability tests, and the identity-guarded
per-instance adapter. No safety or durability code was dropped to hit a line
count.

A leaf that matches the CPU reference does not accept whole-encoder precision, and
no timing claim is made here.