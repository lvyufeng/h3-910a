# R359 — 反汇编终检：高层 `Matmul` 版内核同样**没有 cube 指令**（只有 `LD./ST./ADD.` 标量流）

## 一、本轮静态检查

对新构建的"高层 `Matmul`"kernel（M=16,N=32,K=16）反汇编（3608 行）：

助记符直方图（Top）：

```
MOV 581 / LD. 473 / ADD. 430 / ST. 410 / JUMPC 204 / ZEROEXT. 197 / SIGNEXT. 187 / JUMP 155 / SHL. 118 / CMP.EQ. 89
```

**没有 `MAD.`（cube 乘加）**，也没有 `CBUF/L0A/L0B/L0C` 相关助记符。

## 二、含义

即便是"能算出正确第 0 行"的高层 `Matmul`，其二进制里**也没有 cube 指令**：
说明该路径在本机被编译成**标量/向量模拟**（`LD./ST./ADD.` 序列），而不是调用 cube。
这解释了为什么它只能算出第 0 行——它走的根本不是 cube 通路。

## 三、本季 AscendC 路线的最终结论（证据完整）

| 证据 | 内容 |
|---|---|
| 静态 | 手写 `Mmad` 版 kernel：**0** 条 MAD/CBUF |
| 静态 | 高层 `Matmul` 版 kernel：**0** 条 MAD（仅标量 LD./ST./ADD.） |
| 静态（对照） | CANN 自带 910A `Conv2D`：**144** 条 `MAD.` |
| 编译 | `KERNEL_TYPE_AIC_ONLY` / `MIX_VECTOR_CORE` 均报 **not support in current core version** |
| 数值 | 所有 AscendC 变体：全 0 或"只有第 0 行正确"，与 tiling/格式/参数无关 |

⇒ **结论（终）**：在本机 CANN 9.2.0 + 910A（v200 目标）下，
**AscendC（无论底层原语还是高层 Matmul）都不会生成 cube 指令**，
因此无法用来写 FA 内核。CANN 自家 910A 算子含 MAD，说明它们不是用 AscendC 构建的
（`ops_cv/ascendc/*` 里 `AddConfig("ascend910")` 数为 0 亦印证）。

## 四、唯一可行路径

**TBE/TIK**：CANN 的 910A 算子即此实现；主机上有完整源码
`python/site-packages/te/tik/api/cube/matmul.py`。
验证顺序（新会话执行）：
1. TIK 写 16×16 matmul → **反汇编确认有 `MAD.`** → 再跑数值判据 `A=I ⇒ bad=0`；
2. 通过后移植 64×512×128 FA；
3. 接回 `VLLM_OMNI_910A_FLASH_ATTN` 开关，跑 155 条 Half 检查 + relRMS 门禁。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- DiT ≤10/15 s 未达成；AscendC 路线**证据完整地关闭**；
- 反汇编证据：远端 `/tmp/h3_r221/mm_dis.txt`、`dis.txt`、`cann_mm.txt`。
