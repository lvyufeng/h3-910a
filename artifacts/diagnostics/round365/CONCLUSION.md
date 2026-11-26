# R365 — 定论：本机（CANN 9.2.0 + ascend910/v200 target）**没有任何可用的 cube 代码生成路径**

## 一、决定性证据链（编译器自己的报错给出签名与支持性）

1. `copy_matrix_cc_to_cbuf` 的真实签名由编译器错误直接给出：

```
error: function type 'void (__cbuf__ float *, __cc__ float *, unsigned long, unsigned long) noexcept'
       of 'copy_matrix_cc_to_cbuf' does not support the given target feature
```

⇒ 元数是 **4**（我们试对了），但**该指令不被当前 target 支持**。

2. 编译器内建清单里**根本没有 L0C→GM 的直通指令**：
   - 只有 `copy_matrix_cc_to_ubuf`（L0C→UB）与 `copy_matrix_cc_to_cbuf`（L0C→L1）；
   - 两者在本 target 上都报 "does not support the given target feature"（R363/R365）。

3. 因此：**L0C 里的 cube 结果无法搬出** ⇒ 用户 kernel 无法完成一次 matmul。

## 二、这解释了此前所有现象

| 现象 | 解释 |
|---|---|
| AscendC 手写 `Mmad` 输出全 0 | cube 指令未被生成/不支持，数据从未进入 L0C |
| 高层 `Matmul` 只算出第 0 行、反汇编 0 条 MAD（R359） | 该路径在本 target 被编译成 **标量模拟**（LD./ST./ADD.），不是 cube |
| TIK 能编译到代码生成但 L0C 写回报 UB 不支持（R363） | 同一根因：本 target 无 L0C 写回指令 |
| CANN 自带 910A Conv2D 有 144 条 MAD | 那是**华为用内部/旧工具链**预编译的产物，不代表这套公开工具链能生成 |

## 三、结论（可以下）

**在本机软件栈（CANN 9.2.0，build target = ascend910/v200）上，
用户自定义 kernel 无法生成可用的 cube 代码**：
AscendC 高层/底层、TIK 高层、raw builtin 四条路径均被同一条编译器限制挡住。

⇒ **FA 内核 AscendC/TIK 重写在本机不可行**，DiT ≤15/10 s 的目标因此**在当前软件栈上不可达**。
这不是"参数没调对"，而是 **target feature 缺失**（编译器明确报错）。

## 四、要达成 15/10 s，需要满足其一

1. 升级/更换 CANN，使 `ascend910` target 支持 cube 指令生成（并用 R356/R359 的方法先验证 `MAD` 出现）；
2. 换到 CANN 明确支持的硬件（910B/910_93），先做同样的"反汇编确认 MAD"门槛，再写 FA；
3. 放弃自定义 kernel 路线，接受当前交付（16.34 s/step，−8.9%）作为该机器的上限。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- 本季自定义 kernel 探索到此**有完整证据地关闭**。
