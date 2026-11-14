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
