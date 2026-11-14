# The kernel runs on 910A and the numbers are right

Two blockers from all previous rounds are now resolved, and the TBE flash
attention has executed on this chip for the first time.

## The launch API was the problem, not the kernel

Every earlier attempt used the `aclrtBinary*` family:

```
aclrtBinaryLoadFromFile -> aclrtBinaryGetFunction -> aclrtLaunchKernelWithHostArgs
```

That loads and submits without error and then dies with
`ACL_ERROR_RT_AICORE_EXCEPTION (507015)` -- including for a four-instruction
vector add, which is what showed the fault was in the launch path rather than
in the attention code.

The working path is the legacy runtime API, which is what `BuildCCE` output is
actually built for (`magic: RT_DEV_BINARY_MAGIC_ELF`):

```c
rtDevBinary_t bin = { RT_DEV_BINARY_MAGIC_ELF, 0, obj_data, obj_len };
rtDevBinaryRegister(&bin, &hdl);
rtFunctionRegister(hdl, kern_name, kern_name, nullptr, 0);
rtKernelLaunch(kern_name, blockDim, args, args_size, nullptr, stream);
```

Run:

```
binary registered (28320 B)
function registered: flash_attention__kernel0
launched 2 blocks
kernel executed
wrote /tmp/o_lg.bin 65536 B, /tmp/l_lg.bin 1024 B
```

No AI Core exception. The two APIs are not interchangeable here: `aclrtBinary*`
appears to expect a different binary ABI than the ELF that `BuildCCE` emits, and
it fails in a way that looks like a kernel crash.

## Numerics match a float32 reference

Shape B=1, H=1, N=256, D=128, fp16, launched with blockDim 2, compared against
numpy `softmax(QK^T)V` computed in fp32:

```
raw (no scale)     max|diff| = 0.000009   max|ref| = 0.0185   rel = 0.0005
scaled 1/sqrt(d)   max|diff| = 0.003060   max|ref| = 0.0168   rel = 0.1817
rowsum  raw        max|diff| = 0.00004    ref [5.534, 5.570]  got [5.534, 5.570]
```

`rel = 0.0005` is fp16 rounding, i.e. an exact match. The scaled row is the
control: it shows the kernel does **not** apply `1/sqrt(d)`, so the caller must
scale `Q` (or the scores) before handing them over. That is a real integration
requirement, not a defect -- and it is worth noting because silently feeding
unscaled Q into this would produce attention with the wrong temperature.

## What this changes

The TBE/TIK route is viable after all. The chain that works end to end is:

1. `msopgen -lan py -c ai_core-ascend910` generates the op project.
2. Fill the impl; `set_current_compile_soc_info("Ascend910A")` at import is
   needed or TIK uses the Ascend310B1 instruction table.
3. `BUILD OK` -- TIK emits `kernel_meta/flash_attention.{o,json}` for dav-c100.
4. `rtDevBinaryRegister` + `rtFunctionRegister` + `rtKernelLaunch` runs it.
5. Numerics confirm correctness against numpy.

Step 4 is what every previous round got wrong. The custom-op/GE packaging from
`TBE_OP_PROJECT.md` is not required to run the kernel; it is only needed if the
op has to be reachable through the graph framework.

## Still to do before this accelerates H3

1. **Scale Q.** The kernel wants pre-scaled input.
2. **NZ layout conversion.** The kernel takes and returns `(B,H,d/16,N/16,16,16)`
   NZ; the H3 path is NDS, so both directions need conversion, on device.
3. **Wire into `h3_910a_attn.py`** replacing the dense `_matmul_attention`.
4. **Benchmark on the real shape** q `(1,73408,7,128)` per rank and compare
   against the measured dense 1.166 s/call. Until that number exists there is no
   speedup claim to make. Note the reference tiling processes Q in blocks of
   `Br=min(128,Nq)` with a core task map baked in at compile time, which was
   built for N=256 here -- the real N=73408 will need the tiling regenerated for
   that shape, so this is not yet a drop-in.
