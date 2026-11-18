# Round89 leaf — real vision LayerNorm/GELU controls (source only)

Scope: `artifacts/diagnostics/round89/leaf` only. Owner `vision-leaf89-fresh`
(task34). Nothing here was compiled, launched or executed on an NPU by the author:
no SSH, no NPU/GPU work, no model run, no CANN build, no install, no production
edit, no commit. [Round78 CPU FP16 reference](<../../round78/README.md>) and the
original checkpoint are untouched by this source drop.

## Why these two leaves

R87 ([one-cast biased Half linear](<../../round87/README.md>)) lowered the real
qkv/mlp1/mlp2 CPU-Half error by 31–51×, yet the complete 27-block vision error
only went 1.4568% → 1.2887% at +1% cost. R88
([accurate head_dim=72 attention](<../../round88/README.md>)) lowered its real
SDPA leaf error ~6×, yet the joint full-vision error is 1.1934% at 3.047s versus
1.479s native, so it stays disabled. Both results say the residual is spread out,
not concentrated in the operators already tested.

The two non-linear leaves that were never measured on real operands are the
vision LayerNorm (`_NpuBF16CompatibleLayerNorm`, which for Half input simply calls
`F.layer_norm`) and the two GELUs, which use **different original modes**:
`nn.GELU(approximate="tanh")` in the MLP and exact-erf `nn.GELU()` in the mergers.
Unifying those modes would change the model, so the module pins them per module
name instead.

## Real inventory this module validates (not assumed)

| Item | Count | Detail |
|---|---:|---|
| blocks | 27 | `norm1` + `norm2` each |
| merger modules | 4 | `merger` + 3 `deepstack_merger_list` entries |
| LayerNorm modules | 58 | 55 × `(1152,)`, 3 DeepStack × `(4608,)` |
| GELU modules | 31 | 27 × `tanh`, 4 × `none` (erf) |
| norm eps | — | `1e-6`, untouched |

The seven captured leaves and their real fixture geometry:

| Leaf | Kind | Shape |
|---|---|---|
| `blocks.0.norm1` | LayerNorm | 8064 × 1152 |
| `blocks.0.norm2` | LayerNorm | 8064 × 1152 |
| `blocks.0.mlp.act_fn` | GELU tanh | 8064 × 4304 |
| `deepstack_merger_list.0.norm` | LayerNorm | 2016 × 4608 |
| `deepstack_merger_list.0.act_fn` | GELU erf | 2016 × 4608 |
| `merger.norm` | LayerNorm | 8064 × 1152 |
| `merger.act_fn` | GELU erf | 2016 × 4608 |

## API the driver uses

```python
report = vision_leaf89.run_controls(vision, inputs, compare, tensor_info)
```

`inputs` maps exactly those seven names to the frozen NPU Half tensors captured by
forward pre-hooks. `compare(a, b)` and `tensor_info(x)` are injected by the driver,
so this module never owns a numerical convention and chooses no threshold.

Each leaf reports `native_vs_cpu_half`, `npu_float_op_then_half_vs_cpu_half`,
`cpu_float_op_then_half_vs_cpu_half`, `npu_float_op_vs_cpu_float_op`,
`native_vs_npu_float_op_then_half`, both repeat-identity flags, the input/parameter
SHA checks and `no_accuracy_tolerance`. The `native` reference is taken through
`type(module).forward(module, x)`, so an installed instance patch cannot contaminate
it, and `run_controls` refuses to run if any leaf already carries an instance
`forward`.

`cpu_float_op_then_half` vs `cpu_half` is the interesting control: it *tests* the
usual "Half operator equals Float operator with one final cast" belief instead of
assuming it. Both Float variants derive Float gamma/bias from the original Half
parameters inside the call only. No Float weight is cached or returned, no Float
linear/Conv exists and no BF16 value is produced.

## Optional adapter

```python
adapter = vision_leaf89.VisionNonlinearAdapter89(vision)   # installed, both modes off
adapter.set_modes(norm=True, gelu=True)                    # opt-in, per run
adapter.snapshot()
adapter.uninstall()
```

Only the `forward` attribute of the counted norm and GELU **instances** is wrapped
via `types.MethodType`; `torch`, `torch.nn.functional`, biased linears, the patch
`Conv3d`, attention and the merger linears are untouched. With both modes off the
wrapper calls the captured original bound method, so the disabled path is exactly
the native operator. Enabled LayerNorm is
`F.layer_norm(input.float(), original_normalized_shape, HalfGamma.float(),
HalfBias.float(), original_eps).to(Half)`; enabled GELU is
`F.gelu(input.float(), approximate=original_mode).to(Half)`. eps and
`normalized_shape` are read from the module and never changed. Non-Half or non-NPU
inputs fail closed.

`uninstall()` restores the captured bound methods after checking
`__self__`/`__func__` identity, is idempotent, and refuses to blind-restore a
third-party replacement.

## Non-claims

- No numerical result, no tolerance and no speed claim in this drop.
- A smaller leaf error does **not** make the encoder pass; the parent must still
  run whole-vision, full-encoder and official-task comparisons.
- The adapter is a diagnostic switch, not an accepted optimization.
- `manifest89.json` records hashes of these sources relative to this directory and
  declares `compiled=false`, `executed_on_npu=false`, `proves_precision=false`.