# R354 — 找到 910A 上真正可用的 cube 硬件内建；AscendC 封装可能就是失效层

## 一、本轮事实

1. **先纠正一个我可能过度的结论**：R353 用"L0A 读回全 0"推断 `LoadData` 无效——但
   **L0A→UB 的拷贝本身可能就不被支持**，所以那个回读本身不可靠，不能单独作为证据。
   R353 的结论应降级为"高度可疑"，待下面这条更直接的证据支持。

2. 编译器内建头 `tools/bisheng_compiler/lib/clang/15.0.5/include/cce_aicore_intrinsics.h`
   **确实暴露了 910A 的 cube 硬件原语**：

```
__builtin_cce_load_cbuf_to_ca            → void load_cbuf_to_ca(...)        // L1 → L0A
__builtin_cce_load_cbuf_to_cb            → void load_cbuf_to_cb(...)        // L1 → L0B
__builtin_cce_load_cbuf_to_ca_transpose  → ..._transpose(...)
__builtin_cce_copy_matrix_cc_to_cbuf     → void copy_matrix_cc_to_cbuf(...)  // L0C → GM/L1
__builtin_cce_copy_matrix_cc_to_ub       → void copy_matrix_cc_to_ub(...)    // L0C → UB
```

⇒ **硬件通路在编译器层面是完整可用的**，与框架 Matmul 能算出结果一致。

3. 这些声明签名都是 `(...)` 可变参（无类型检查），**参数顺序必须从 TIK 侧源码取得**。
   TIK 的 Python 实现在主机上：`/home/mseco/Ascend/cann-9.2.0/python/site-packages/tbe/`（含 `tik/`），
   它内部正是把 `tik.load`/`tik.mmad` 映射到这些 builtin——**这就是权威的参数顺序表**。

## 二、推断

AscendC 的 `LoadData`/`Mmad` 封装在 **v200（910A）** 上很可能没有正确映射到这些 builtin
（或映射到了不同语义），而框架 `Matmul` 走的是另一条内部路径，所以能算。
这也解释了为什么换遍所有 AscendC 参数都毫无变化。

## 三、下一步（最后一条路，且是权威路径）

1. 读 `python/site-packages/tbe/tik/`（或 `tbe/dsl/`）里 `load`/`mmad` 的实现，
   抄出它们调用 `__builtin_cce_load_cbuf_to_ca` / `mmad` 的**确切参数顺序**；
2. 在 AscendC kernel 里直接调用这些 builtin（`load_cbuf_to_ca(...)` 等，头文件已包含）；
3. 判据：A=I 时 `bad=0`；
4. 若 builtin 路径成功 ⇒ 用 TIK 或直接 builtin 重写 FA（这才是 910A 的正路）。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：框架级 cube 可用；**硬件 builtin 已确认存在**；AscendC 封装层高度可疑（非定论）；
- 工程：`/tmp/h3_r221/`（含 msopgen 工程与判据 harness）；
- 远端 master 含 R332→R354 全部记录。
