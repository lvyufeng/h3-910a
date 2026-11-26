# R369 — TIK FA 内核：流水线、基线性能与 tile 上限（走 TIK 路线的起点）

## 一、部署链路（已确认）

```
vllm_omni/diffusion/attention/backends/h3_910a_attn.py
  └─ 若 VLLM_OMNI_910A_FLASH_ATTN=1 且形状合格
       └─ fa910a.flash_attention_910a(q,k,v,n_pad=...)      # ctypes 包装
            └─ /tmp/libfa910a.so (fa910a_launch)             # 自建 C launcher
                 └─ /tmp/fa910a_kernels/<key>/<key>.o        # TIK 编译产物
```

- 内核源码：`/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py`（538 行，标准 TIK FlashAttention-2 实现）
- 构建入口：`/tmp/fa910a_sweep.py::_build()`（`FlashAttentionFwd(...).compute_process()` → `kernel_meta/<key>.o`），单形状构建 ~19 s
- 形状键：`b{b}_h{h}_n{n}_d{d}_br{br}_bc{bc}_c{core}_k{key_limit}_v{n}`，缓存于 `/tmp/fa910a_kernels/`
- tile 上限由环境变量控制：`H3_FA_BR_CAP` / `H3_FA_BC_CAP`（默认 128/304，交付配置 64/912）

## 二、本轮实测基线（B=1, H=7, D=128, fp16）

| N | tile (Br/Bc) | 内核实际 (Nq/Bc/Tc/core) | 中位耗时 |
|---|---|---|---|
| 41792 | 64 / 912 | Nq=43776, Bc=912, Tc=48, core=30 | **137.79 ms** |
| 41792 | 64 / 1024 | 同上 | 137.76 ms |
| 41792 | 64 / 608 | Nq=41952 | 144.61 ms |
| 41792 | 128 / 512 | — | 142.02 ms |
| 41792 | 128 / 304 | — | 154.54 ms |
| 73408 | 64 / 912→608 | Nq=75392, Bc=608, Tc=124 | 427.88 ms |
| 73408（无 trim/默认 tile） | 128 / 304 | — | 452.6 ms |

**模型相关性**：DiT 每步 49 次 attention × 137.8 ms ≈ **6.75 s/step**，与 R126 实测 FA 7.454 s 一致（差额为调度开销）。

## 三、Tile 空间已穷尽（内存硬上限）

| 组合 | 失败原因 |
|---|---|
| 128 / 912 | `C_l0c` 需 466944B > L0C 可用 262144B |
| 256 / 256 | `C_ub` 需 131072B > UB 可用 125568B |
| 32 / 2048 | `Vj_l1_zN` 需 466944B > L1 可用 440320B |

⇒ 在**当前内核的内存使用方式**下，64/912（L0C: 64×912×4 = 233472B ≤ 262144B）就是可行域最优。

## 四、效率与提升空间

- 计算量：4·H·N²·D（N=41792, H=7, D=128）≈ **6.87 TFLOP**
- 实耗 137.8 ms ⇒ **≈49.8 TFLOPS ≈ 19.5% MFU**（910A FP16 峰值 256 TFLOPS）
- 若要 FA 降到 ~50 ms（10 s/step 所需量级），需要 **~100 TFLOPS（39% MFU）**——即内核效率翻倍

## 五、下一步（按性价比排序）

1. **UB 占用优化**：当前 softmax 的 S tile 用满 UB；若改成分段 softmax（S 分两半过 UB），可释放 UB 给更大 Br，属内核级改动；
2. **K/V 双缓冲**：检查 `compute_in_each_kv_block` 中 K/V 的 L1 装载是否单缓冲，若是则改双缓冲以掩盖 MTE2 延迟；
3. **减少 ones-matmul 行和**：当前用 L1 全 1 张量做 cube 行和，可评估用向量归约替代（UB 允许时）；
4. 若上述收益 <10%，则该内核对 910A 已接近实际上限，10 s/step 需另寻路线（如非 FA 区域 + 更强硬件）。

## 六、可复用的迭代工具

```bash
# 单形状 FA 计时（本会话新增）
FA_N=41792 H3_FA_BR_CAP=64 H3_FA_BC_CAP=912 \
H3_FA215_KERNEL_DIR=/tmp/fa910a_kernels H3_FA215_PKG=/tmp/fa_pkg \
/home/mseco/vllm910a/bin/python -u /tmp/h3_r221/fa_bench_now.py /tmp/out.json
```
新 kernel 首次调用会自动构建（~19 s），之后走缓存。
