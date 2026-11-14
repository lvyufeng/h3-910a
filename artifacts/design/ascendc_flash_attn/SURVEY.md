# Survey: public Ascend 910A code for flash attention

Searched GitHub for 910A / AscendC / flash-attention material (16 repo queries,
deduplicated). The headline result is a negative one that changes the plan:
**no public AscendC flash attention exists for 910A.** Everything that is
focused and maintained targets 910B/910C/950.

## What targets 910A (dav-c100)

| Source | What it gives | Limitation |
|---|---|---|
| `flash_attention-1.0.0-py3-none-any.whl` (TIK/TBE) | complete FA2 forward + backward, tiling, online softmax | TBE/TIK DSL, not AscendC; needs porting |
| `monellz/ascend910a-extras` | `matmul_core.h` cube plumbing, `paged_attention_ex.cpp` scaffold, `msopgen` build path with `-c ai_core-ascend910` | attention softmax body is commented-out pseudo-code |

Those two remain the only 910A references, and they are complementary: one has
the algorithm, the other has the AscendC cube contract and a build path.

## What excludes 910A

| Repo | Explicit statement | Verdict |
|---|---|---|
| `MinghuasLab/flash-attention-npu` (97★) | setup.py: `"v2" ... (910B/C only)`, `"910" only Ascend 910B/C backends`; README: "Hardware: Ascend 910B / 910C / 950" | ✗ |
| `zhutianci/ascendc_kernel` | "昇腾 910B4（20 AIC + 40 AIV，CANN 9.1.0）" | ✗ |
| `374954417/ascend-910b-irev` | "910B(dav-c220 / Ascend910_9382)" | ✗ |
| `HyosungSink/CANN-AscendC-optimization-Cases` | "Ascend A2（910B）与 A3" | ✗ |
| `BobbyWeng/flash-attention-ascendc-costmodel` | `arch22` / `arch35` (910B / 910C) | ✗ |
| `dzcmingdi/flash-attention-ascend` | TBE, `ai_core/ascend310` config, 2023 | ✗ |

Note the trap: `MinghuasLab/flash-attention-npu` has a `csrc/ascend910/`
directory, which looks like 910A. It is not — in this repo "ascend910" is the
910B/910C backend name, and `setup.py` says so outright. The directory name is
not evidence of A-generation support. The same ambiguity applies to
`msopgen -c ai_core-ascend910`, which serves both.

## Still worth borrowing

Even though these do not run here, two are structurally valuable.

**`MinghuasLab/flash-attention-npu`, `csrc/ascend910/flash_attn_npu/`** is the
best available template for how a real AscendC flash attention is factored:

```
online_softmax.hpp    qk_matmul.hpp      pv_matmul.hpp
rescale_o.hpp         copy_l1_to_l0.hpp  fag_mmad_cube1/2/3.hpp
fag_common/cube_addr.h  fag_common/vector_addr.h
fag_tiling.cpp        softmax_tiling.cpp  fag_kernel_launch.hpp
```

Three successive kernels (`flash_attn_npu`, `_3`, `_4`) show what actually had
to change between attempts — useful as a map of the hard parts, above all the
cube/vector handoff and the tiling, which are the same problems on 910A.
The primitives are arch-specific and will not transfer; the decomposition will.

**`374954417/ascend-910b-irev`** reverse-engineers dav-c220 instruction
encoding by single-instruction kernels plus byte-level diffs. The arch differs
(dav-c220 is not our dav-c100), so the encodings are not reusable, but the
*method* is, and it is the only documented route to hand-verifying what the
assembler emitted.

## A design constraint worth recording

`zhutianci/ascendc_kernel` notes that Fixpipe writing straight to UB is only
supported on Atlas 200I / 500 A2 **inference** products, not on A2 training
cards. Since 910A is a training part, the kernel must not assume a direct
L0C -> UB fixpipe. That matches `matmul_core.h`, whose path is always
CO1 -> CO2 -> Cast -> GM/L1, never L0C -> UB directly. The draft kernel's
`Fixpipe`-to-UB assumption is therefore wrong for this chip and has to go
through CO2.

## Consequence for the plan

Three options, in order of cost:

1. **Port the TBE reference's algorithm onto the `extras` cube contract.** Both
   are 910A, so the pieces fit; this is the intended route and what
   `DESIGN.md` describes. Cost: write the vector softmax (no reference has it
   for 910A), then `msopgen`, compile, validate.
2. **Run the TBE kernel as-is.** It is a TBE DSL implementation, so CANN must
   still compile it for dav-c100 — but TBE compilation happens at op build time
   and 910A TBE is exactly what our CANN 9.2.0 supports. This may be cheaper
   than a hand port and is worth testing *before* writing AscendC: if
   `flash_attention-1.0.0` can be registered and run, the task collapses from
   "write a kernel" to "wire up an existing one".
3. Fall back to the current dense path (448 s, 4 steps).

Option 2 should be tried first. It is a small experiment with a large payoff,
and it is the one path where the expensive part already exists for this chip.
