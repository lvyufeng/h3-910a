# The TBE flash attention compiles for 910A

Result: **`flash_attention-1.0.0`'s TIK kernel builds for `dav-c100` on our
CANN 9.2.0.** No AscendC port is needed to get a working kernel; the existing
implementation is enough.

## The one thing that blocked it

TIK picks its target SoC from `tbe.common.platform`, and on this install the
default is **wrong**:

```
soc before: Ascend310B1          # default, nothing to do with our chip
soc after : Ascend910A           # after set_current_compile_soc_info
```

Building without fixing that raises a misleading error deep inside the kernel:

```
Error: Ascend310BAiCore Instruction tensor_mov doesn't support cc to ubuf.
        tik_ops_utils.py:356  tensor_mov(C_ub, C_l0c, "m", ...)
```

The message reads like a hardware limitation of our chip. It is not. It is
TIK validating L0C -> UB against the **Ascend310B** instruction table because
it thinks the target is a 310B. This is the same class of failure as the
`RmsNorm ... no supported kernel` incident earlier in the project, where the
real cause was a missing `cstdint` include and the message blamed the op.

Fix, one line before building:

```python
import tbe.common.platform as plat
plat.set_current_compile_soc_info("Ascend910A")
```

## Verified platform values

With the SoC set, `get_soc_spec` reports the hardware we actually have, and
these are worth keeping because they bound any future kernel:

| spec | value |
|---|---|
| `CORE_NUM` | 32 |
| `L1_SIZE` | 1 048 576 (1 MiB) |
| `L0C_SIZE` | 262 144 (256 KiB) |

Note `Ascend910A.ini` does exist in `aarch64-linux/data/platform_config/`
(73 files there, alphabetically the 950 and Kirin entries come first, which is
why it looks absent on a casual listing).

## Artefacts produced

```
/tmp/kernel_meta/flash_attention.cce    277 512 B   generated CCE source
/tmp/kernel_meta/flash_attention.o       28 320 B   ELF, arch 0x1029 (AI Core)
/tmp/kernel_meta/flash_attention.json       605 B   kernelName=flash_attention__kernel0,
                                                    coreType=AiCore, blockDim=2
```

`blockDim=2` because the probe used B=1, H=1, N=256 -> only 2 Q blocks. The H3
shape (B=7 heads, N=73408) generates hundreds and fills all 32 cores.

## How the driver calls it

`compute_process()` calls `BuildCCE` itself, so the public entry builds:

```python
from flash_attention.ops.flash_attention.flash_attention_fwd import flash_attention
q = {"shape": [B, H, d/16, Nq/16, 16, 16],   # NZ: d-fractals outer
     "ori_shape": [B, H, Nq, d],             # ND
     "dtype": "float16", "format": "NZ"}
...
flash_attention(query=q, key=k, value=v, attn_mask=None,
                dropout_mask=None, alibi_mask=None,
                output=o, rowsum=rs, kernel_name="flash_attention")
```

The 6-D NZ shape decodes as `(B, h, N1, M1, M0, N0)` with `d = N1 * N0`, so
`N1 = d/16` and the head-dim fractals are the outer axis.

## What is still unproven

Compiling is not running. Three things remain:

1. **Launch.** The kernel has to be invoked — either through the ACL op
   framework (needs `op_info_cfg` + aclnn registration) or by launching the
   `.o` with the runtime. Nothing here has executed on the device yet.
2. **Numerics.** The reference has never been checked against a reference
   attention on this hardware.
3. **The MindSpore wrapper is unusable.** `flash_attention_impl.py` registers
   through `mindspore.ops.TBERegOp`, and MindSpore is not installed. The build
   above bypasses it by calling the `flash_attention()` builder directly, which
   is the right approach anyway since H3 needs this from torch.

Also note the op declares `F16_FracNZ` for q/k/v, so callers must supply
fractal-NZ tensors; the H3 attention path currently has them in NDS.

---

# It compiles but faults on device

Follow-up after building: the kernel loads and launches, then dies on the
AI Core.

## Launch path works

`aclrtBinaryLoadFromFile` -> `aclrtBinaryGetFunction` -> 
`aclrtLaunchKernelWithHostArgs` all return success, so the compiled `.o` is a
valid binary and the launch plumbing is correct. The kernel signature, read off
the generated source, is

```c
extern "C" __global__ __aicore__ void flash_attention__kernel0(
    __gm__ half* Q_gm, __gm__ half* K_gm, __gm__ half* V_gm,
    __gm__ half* O_gm, __gm__ float* l_gm)
```

five GM pointers. The core task map is baked in at compile time (the `.cce`
writes `core_idx_to_batch_info`/`core_idx_to_tr_info` from immediates at lines
900-907), so nothing has to be passed from the host for that.

## Then the device faults

```
aclrtSynchronizeStream -> 507015   ACL_ERROR_RT_AICORE_EXCEPTION
Message info[0]: RTS_HWTS: aicore exception, slot_id=29, stream_id=77
```

## Root cause: the original error was telling the truth

Before forcing the SoC, the build failed with

```
Error: Ascend310BAiCore Instruction tensor_mov doesn't support cc to ubuf.
        tik_ops_utils.py:356   tensor_mov(C_ub, C_l0c, "m", ...)
```

I read that as TIK using the wrong instruction table and forced
`set_current_compile_soc_info("Ascend910A")` to get past it. The compile then
succeeded -- but forcing the SoC only **bypassed a check**; it did not make the
emitted instruction legal. `tensor_mov` from L0C to UB is the problem, and it is
unsupported on this part.

This matches a constraint already recorded in `SURVEY.md`: Fixpipe writing
straight to UB is an inference-product feature (Atlas 200I / 500 A2), not
something training cards do, and 910A is a training part.

The corroboration is that `matmul_core.h` from `ascend910a-extras` -- code that
does run on this chip -- never goes L0C -> UB. Its path is
`CO1 -> CO2 -> Cast -> GM/L1` throughout.

So: **the `q2374556230`/TBE wheel is written for an inference product and is not
usable on 910A as-is.** It compiles because we silenced the check.

## What this does and does not establish

Verified:

1. `set_current_compile_soc_info("Ascend910A")` makes TBE target dav-c100
   (CORE_NUM 32, L1 1 MiB, L0C 256 KiB).
2. The build produces a real object (277 KB `.cce`, 28 KB ELF with arch
   `0x1029`, json naming `flash_attention__kernel0`).
3. The ACL kernel-launch API works here and needs no MindSpore, so once a
   correct kernel exists it can be driven from this harness.
4. The kernel faults at runtime with an AI Core exception.

Not established: numerics, and whether the fault is *solely* the L0C->UB path
or also something else (UB budget, address assumptions). A crash tells us it is
broken, not which single line broke it.

## Consequence

Two viable routes, both larger than "run the existing kernel":

* **Patch the TBE source.** `matmul_compute` (tik_ops_utils.py:331) and
  `row_sum_cube_impl` (:304) are the two places that do `tensor_mov` L0C -> UB.
  Route them through CO2 then load from L1 into UB, as `matmul_core.h` does.
  The rest of the algorithm is untouched, so this is the cheaper repair and it
  is the only route that keeps the FA2 implementation that already exists.
* **Hand-write AscendC** against `matmul_core.h`'s contract, per `DESIGN.md`.

The first is now clearly better value: the algorithm, tiling and online softmax
are all present and correct-looking, and the defect is localised to two helpers.

---

# Patching the two helpers does not work: 910A TBE has no tensor_mov out of L0C

Tried the repair proposed above -- route the fixpipe through L1OUT instead of UB.
All three `tensor_mov(C_ub, C_l0c, ...)` sites now go via L1OUT
(`tik_ops_utils.py` 362, 427, 485), zero L0C -> UB calls left in the file.

Rebuild fails:

```
Error: Ascend910AiCore Instruction tensor_mov doesn't support cc to cc.
        tik_ops_utils.py:362   tensor_mov(C_l1out, C_l0c, ...)
```

So on Ascend910A the `tensor_mov` primitive refuses **both** directions out of
L0C:

| move | verdict |
|---|---|
| `tensor_mov` L0C -> UB | `doesn't support cc to ubuf` |
| `tensor_mov` L0C -> L1OUT | `doesn't support cc to cc` |

This changes the shape of the problem. The earlier plan assumed a localised
defect -- one unsupported edge -- that a two-line redirect to L1OUT would fix.
It is not that. The `matmul_compute` epilogue in this codebase is built on a
`tensor_mov`-out-of-L0C pattern that 910A's TBE does not provide in any form
tried. Note these are *compile-time* rejections by TIK's own instruction table,
which is the authoritative statement of what this backend can emit.

A correct epilogue still has to exist: `matmul_core.h` does it as
`CO1 -> CO2 -> Cast -> GM/L1`, using `DataCopy` with `BLOCK_MODE_MATRIX` rather
than `tensor_mov`. So the fix is not a redirect but a rewrite of the
matmul-output path in TBE terms, i.e. the same work as writing the epilogue
from scratch, at which point keeping the rest of the TIK algorithm saves less
than originally hoped.

## Honest status

What is established:

1. TBE targets dav-c100 with `set_current_compile_soc_info("Ascend910A")`
   (CORE_NUM 32, L1 1 MiB, L0C 256 KiB).
2. A kernel object is produced and the ACL launch path works with no MindSpore
   (`BinaryLoadFromFile` -> `BinaryGetFunction` -> `LaunchKernelWithHostArgs`).
3. The unmodified reference faults on device with
   `ACL_ERROR_RT_AICORE_EXCEPTION (507015)`.
4. `tensor_mov` is unusable out of L0C on this backend in both directions, so
   the reference's epilogue cannot be repaired by redirecting it.

What is therefore true: **option 2 -- "just run the existing TBE kernel" -- is
closed.** It is not a small wiring job; the primitive it depends on does not
exist here.

Remaining routes, in order of confidence:

1. **Rewrite the matmul epilogue in TBE** using the L0C -> L1OUT path that
   `matmul_core.h` demonstrates, keeping the rest of the TIK algorithm
   (online softmax, tiling, cube rowsum). Medium effort, and the pieces that
   are most likely to be correct (the FA2 maths) are already written.
2. **Hand-write AscendC** against `matmul_core.h`, per `DESIGN.md`. Best
   understood route since that code is known to run on this chip, but the most
   work.
3. Give up on the fused kernel and keep the dense path (448 s, 4 steps).

Nothing here has been numerically validated, and no timing has been measured for
a working kernel, so the 297 s projection in the notes remains unconfirmed.

## Lesson worth keeping

Twice now a TIK error message has been read as evidence about the *hardware*
when it was evidence about *the SoC the backend had selected*:

* `RmsNorm ... no supported kernel` -> actually a missing `cstdint` include.
* `tensor_mov doesn't support cc to ubuf` on Ascend310B -> looked like TIK using
  the wrong table, and forcing Ascend910A did silence it; but the follow-up
  shows 910A rejects the neighbouring form too, so the message was substantively
  right and forcing the SoC only moved the failure later.

---

# RETRACTION: the previous section is wrong. tensor_mov L0C -> UB is legal here.

The conclusion above -- that 910A's TBE has no `tensor_mov` out of L0C, and that
option 2 is therefore closed -- is **wrong**. Retracting it.

## What the authoritative table actually says

`tbe/tik/tik_lib/tik_data_move_api/tik_data_move_common.py:51` holds
`archversion_scope`, the list of scope pairs each arch may move between. For
`ASCEND_910AIC` it contains, among others:

```
'cc_m162ubuf', 'cc_m322ubuf', 'cc_v162ubuf', 'cc_v322ubuf'
```

Those are L0C -> UB, in the matrix (m) and vector (v) modes at 16- and 32-bit.
So L0C -> UB **is** supported on this backend.

The scope string is built by `get_scope_str` (:288):

```python
if src_scope == "cc":
    src_scope = src_scope + "_" + block_mode + str(get_bit_len(src.dtype))
```

so `tensor_mov(C_ub, C_l0c, "m", ...)` with an fp32 L0C source produces
`cc_m322ubuf`, which the table allows. Checked directly rather than reasoned
about:

```
SCOPE_MAP cc = cc   SCOPE_MAP ubuf = ubuf   SCOPE_MAP L1OUT = cc
get_bit_len(fp32) = 32
L0C->UB(f16) 'm'   scope_str=cc_m322ubuf    allowed=True
L0C->UB(f32) 'm'   scope_str=cc_m322ubuf    allowed=True
L0C->L1OUT   'm'   scope_str=cc_m322cc_m32  allowed=False
```

## Why my patch failed, and what it means

`SCOPE_MAP` maps `L1OUT` to **`cc`**, not to `cbuf`. So redirecting to L1OUT
produced `cc_m322cc_m32` -- an L0C-to-L0C move -- and TIK rejected it with
"doesn't support cc to cc". The error was correct and was about my patch, not
about the hardware. I misread it as further evidence that the epilogue was
impossible, when it was evidence that my redirect was invalid.

After reverting all three sites to the original
`tensor_mov(C_ub, C_l0c, "m", ...)`, the build compiles again ("OK: kernel
compiled"). So:

* The original code is **legal** and compiles for 910A.
* The runtime `ACL_ERROR_RT_AICORE_EXCEPTION (507015)` is therefore **a
  different bug**, not an unsupported instruction.
* Option 2 is **not** closed.

## Corrected status of the runtime fault

Still unexplained. The kernel is legal, compiles, launches, and faults on the
AI Core. Candidate causes, none investigated yet:

1. **UB budget.** The generated `.cce` places `core_idx_to_tr_info` at UB offset
   111872 B; the full UB footprint of the S2 path may exceed what this part
   provides at runtime even though TIK accepted it.
2. **Atomic-add outputs.** The reference declares `O_gm` and `l_gm` with
   `is_atomic_add=True`; if the runtime path does not set up atomic
   accumulation the way the kernel expects, the write would fault.
3. **My harness.** Buffer sizes, the per-argument pointer indirection in
   `aclrtLaunchKernelWithHostArgs`, or the absence of an explicit context are
   all unverified.
4. **Genuine kernel bug** for this shape that only shows at runtime.

The next step is to separate (3) from the rest, because a harness bug is both
the most likely and the cheapest to exclude -- for instance by launching a
trivial known-good kernel through the same `harness.cpp` path first. Until that
is done, no conclusion about the TBE kernel's correctness on 910A is warranted.

## Process note

Three turns in a row now, an error message has been read as a fact about the
hardware and used to close a route:

* `RmsNorm no supported kernel` -> missing `cstdint` include.
* `tensor_mov doesn't support cc to ubuf` (as Ascend310B) -> soC default was
  wrong; forcing Ascend910A made it compile.
* `tensor_mov doesn't support cc to cc` -> my own invalid redirect, not a
  hardware limit. And it caused me to wrongly retract a working approach.

The pattern is that TIK reports the *symptom nearest to the check that fired*,
which may be a consequence of my own edit, of a wrong SoC, or of a build config
problem. Before treating any of these as a hardware fact, verify the claim
against `archversion_scope` / the SoC spec directly. That check is cheap and
would have saved all three detours.
