# The TBE flash attention is now a registered custom op for ascend910

This closes the packaging gap that `TBE_BUILD.md` ended on: a bare
`kernel_meta/*.o` is not launchable, and the supported route is to package the
TBE implementation as a custom operator so the framework supplies workspace,
tiling and atomic setup at launch.

## msopgen generates a TBE op project, not only AscendC

The key unknown last round was whether this CANN still ships the TBE op flow.
It does. `msopgen gen -lan py` generates a **TBE** project (`-lan py` is
documented as "dsl and tik coding language"; `-lan cpp` is the AscendC one):

```bash
msopgen gen -m 0 -i op_def.json -f aclnn -c ai_core-ascend910 -lan py -out ./out
```

What it produces, and note the SoC-specific config directory:

```
out/tbe/op_info_cfg/ai_core/ascend910/flash_attn_ex.ini
out/tbe/impl/flash_attn_ex.py        <- fill this in
out/op_proto/flash_attn_ex.{h,cc}
out/op_tiling/ out/framework/ out/cmake/ out/scripts/ out/build.sh
```

## Build

One undocumented prerequisite: `build.sh` insists on
`ASCEND_TENSOR_COMPILER_INCLUDE` and looks for `/usr/local/Ascend/compiler/include`,
which does not exist on this install. The header root that does exist, and that
contains `graph/types.h`, is `$ASCEND_HOME_PATH/include`:

```bash
export ASCEND_TENSOR_COMPILER_INCLUDE=/home/mseco/Ascend/cann-9.2.0/include
bash build.sh
```

Result:

```
[100%] Built target custom_opp_ubuntu_aarch64.run
[INFO] Build successfully, the target is in .../build_out
```

The package carries `vendors/customize/op_proto/libcust_op_proto.so`, the TBE
impl, and the per-SoC config.

## Install

```bash
bash custom_opp_ubuntu_aarch64.run --quiet
# install package to .../cann-9.2.0/opp/vendors ... SUCCESS
```

Installed layout:

```
opp/vendors/customize/op_impl/ai_core/tbe/config/ascend910/aic-ascend910-ops-info.json
opp/vendors/customize/op_impl/ai_core/tbe/customize_impl/flash_attn_ex.py
opp/vendors/customize/op_impl/ai_core/tbe/customize_impl/flash_attention/...
opp/vendors/customize/op_impl/cpu/aicpu_kernel/
opp/vendors/customize/op_proto/
```

## Registration verified

`aic-ascend910-ops-info.json` contains the operator, and this is the evidence
that it is now known to the framework for our SoC rather than sitting as a loose
object file:

```json
"FlashAttnEx": {
  "input0":  {"dtype": "float16", "format": "ND", "name": "q"},
  "input1":  {"dtype": "float16", "format": "ND", "name": "k"},
  "input2":  {"dtype": "float16", "format": "ND", "name": "v"},
  "output0": {"dtype": "float16", "format": "ND", "name": "o"},
  "opFile":      {"value": "flash_attn_ex"},
  "opInterface": {"value": "flash_attn_ex"}
}
```

## The impl

`flash_attn_ex.py` wraps the reference. It pins the SoC before building, since
the default here is `Ascend310B1` and that silently selects the wrong TIK
instruction table, and it converts the ND `(B,h,N,d)` descriptors to the 6-D NZ
shape `(B,h,d/16,N/16,16,16)` the kernel expects:

```python
set_current_compile_soc_info("Ascend910A")
flash_attention(query=_prep(q), key=_prep(k), value=_prep(v),
                attn_mask=None, dropout_mask=None, alibi_mask=None,
                output=_prep(o), rowsum=_prep(l, with_nz=False),
                tiling_stgy_name="sparse", kernel_name=kernel_name)
```

## What this does and does not achieve

Done and verified:

* A TBE op project for `ai_core-ascend910` is generated, **built** and
  **installed**.
* `FlashAttnEx` is registered for our SoC with the expected signature.

Still open:

1. **No aclnn interface was generated.** `-f aclnn` did not emit aclnn stubs;
   the package contains only `libcust_op_proto.so` plus the TBE impl. So this
   cannot yet be called the way the H3 path would want it — a direct
   `aclnnFlashAttnEx` call. Invoking it now requires going through the GE graph
   API, or writing an aclnn wrapper by hand.
2. **The op has not been executed.** No numerics, no timing. Whether the TBE
   kernel runs correctly on 910A is still unknown; only its packaging is
   proven.
3. The registered `output1` (`l`, fp32) is absent from the ops-info entry even
   though the `.ini` declared it, so the output arity may need reconciling
   before a graph call will admit the op.

## Next, in order

1. Invoke `FlashAttnEx` once through a GE graph with a small shape and compare
   against a numpy reference. That finally answers whether the reference kernel
   is correct on this chip, which is the question every earlier attempt
   deferred.
2. If it runs, generate or hand-write the aclnn wrapper so vllm-omni can call it
   instead of going through a graph.
3. Only then measure performance and decide whether the fused path beats the
   current dense one.
