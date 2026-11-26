# R364 — raw builtin 能编译通过，但 MAD=0（结论**受死代码消除干扰，不可下结论**）

## 一、本轮实验

在 AscendC kernel 里直接调用编译器内建（`cce_aicore_intrinsics.h`）：

```cpp
__cbuf__ half *pa1 = (__cbuf__ half *)a1.GetPhyAddr();   // L1
__ca__  half *pa0 = (__ca__  half *)a0.GetPhyAddr();     // L0A
__cb__  half *pb0 = (__cb__  half *)b0.GetPhyAddr();     // L0B
__cc__  float *pc0 = (__cc__ float *)c0.GetPhyAddr();    // L0C
DataCopy(a1, qGm, 256); DataCopy(b1, kGm, 256);
load_cbuf_to_ca(pa0, pa1, 0, 1, 0, 0, false);
load_cbuf_to_cb(pb0, pb1, 0, 1, 0, 0, false);
mad(pc0, pa0, pb0, 16, 16, 16, 0);
```

- **编译通过**（`build_rc=0`）——说明 raw builtin 在 AscendC kernel 里**可用**（不再被 API 层拒绝）；
- 但反汇编 **`MAD` 计数 = 0**。

## 二、为什么这个 0 **不能**作为结论

本轮 kernel **删掉了写回**（`copy_matrix_cc_to_cbuf` 因参数元数未知、`too many` 编译失败，被临时移除），
因此 `load_*`/`mad` 的结果**没有任何 observable 副作用** ⇒ 编译器做**死代码消除**完全合理。
**MAD=0 很可能是被优化掉，而不是不发射。**

（此前 R356/R359 的 "0 条 MAD" 不受此影响：那两版 kernel 有完整的 C 写回与输出，代码必然是活的。）

## 三、下一步（明确）

1. 先确定 `copy_matrix_cc_to_cbuf` 的**真实参数元数**（试 2/3/4 个参数；TIK 生成的
   `copy_matrix_cc_to_ubuf` 是 8 参，但 cbuf 变体不允许 6 参以上，说明签名不同）；
2. 补上写回、让 `mad` 结果可达 GM；
3. 再反汇编判 `MAD`；若仍为 0 ⇒ 才能说"raw builtin 在该 target 也不发射"。

## 四、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- **kernel 侧的 raw builtin 路径已被证明可编译**（这是新信息）；
- 判据：`MAD` 出现 + A=I ⇒ C=B；当前两项都未达成，且本轮的 MAD=0 不可采信。
- 工程：`/tmp/h3_r221/faA/proj/op_kernel/flash_attention_custom.cpp` 为 raw builtin 版。
