# R324 — warm 口径复测（同 engine 连跑两趟）：已启动，encoder 复现 62.15 s

## 一、设计

为拿到与既有收据可比的 **warm VAE** 数字（目标 video VAE < 20 s），写 `warmvae324.py`：

1. 复用官方启动前言：`H3_R126_GENERATION_OUT` + `from full_bootstrap126 import install_alignment`
   + `torch.npu.matmul.allow_hf32=False`（缺一即 `SetPrecisionMode 500001` / bootstrap 断言）；
2. 一个 engine 内**连跑两趟** `omni.generate`（同 seed、同参数）；
3. 探针（`sitecustomize` 惰性 patch）在两趟里各打印一次 `encode_prompt` / `decode_latent`；
   ⇒ **第 1 趟 = cold VAE，第 2 趟 = warm VAE**，同一进程同一权重驻留状态。

## 二、踩坑

- 模块顶部直接 `mkdir(exist_ok=False)` → **multiprocessing spawn 会重新执行模块**，子进程 mkdir 失败；
- 修法：全部逻辑包进 `main()` + `if __name__ == '__main__'` 守卫（与官方脚本一致）。

## 三、已得数据

```
[STAGE] encoder 62.141–62.148 s   (8 个 TP8 worker)
```

⇒ 与 R321 的 **62.95 s** 相差 1.3%，**encoder 方法级耗时稳定可复现**。

## 四、待完成

- 第 1 趟 cold `decode_latent`（预期 ~80 s video + ~60 s audio，与 R323 一致）；
- 第 2 趟 warm `decode_latent`（这是与 **< 20 s** 目标直接对比的数字）；
- 两趟 wall、以及 warm 值是否仍 ~20.5 s。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，`H3_PERF_COMBO=1` 默认 OFF；
- ts-217 健康；15 s 目标：R313 路线级结论 ⇒ 当前条件不可达。
