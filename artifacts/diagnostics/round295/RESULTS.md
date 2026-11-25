# R295 — S1a-0 通过：自定义 FA 算子在 **910A** 上经 aclnn 真实执行

## 一、做了什么

1. 按 **`ai_core-Ascend910A`** 重新生成 FA 算子（R294 的 SoC 修正）；
2. 写第一个最小 kernel（`op_kernel/flash_attention_custom.cpp`，**identity：o = q**）：
   `TPipe` + `TBuf<VECCALC>` + `DataCopy(GM→UB)` + `PipeBarrier` + `DataCopy(UB→GM)`，仅 block 0 执行；
3. `bash build.sh` → `custom_opp_ubuntu_aarch64.run` 打包成功；
4. 安装到 `/tmp/h3_r221/oppFA`，`source vendors/customize/bin/set_env.bash`；
5. 写 C++ aclnn 宿主 `test_fa294.cpp`（已入库 `round295/deployed/`）：
   `aclCreateTensor`(2-D 64×128, ND) → `aclnnFlashAttentionCustomGetWorkspaceSize(q,k,v,o,&ws,&ex)`
   → `aclnnFlashAttentionCustom(ws,ws,ex,stream)` → sync → 回读比较。

## 二、实测结果（通过）

```
tensors q=0x… k=0x… v=0x… o=0x…
workspace=0
RESULT M=64 K=128 maxdiff_vs_q=0 bad=0 head=(0.4231,0.4939,0.3970)->0.4231
```

- **`maxdiff_vs_q = 0`、`bad = 0`** ⇒ kernel 真的在设备上跑了，输入经 UB 中转后逐位写回输出；
- 这证明 **msopgen → 910A 编译 → 安装 → aclnn 两阶段调用 → kernel 执行 → 显存读写** 整条链路可用。

## 三、接口现状（重要）

生成的 aclnn 签名为 **3 输入 + 1 输出、无 attr**：

```c
aclnnStatus aclnnFlashAttentionCustomGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
```

- 我们的 `fa.json` 里 `attr_desc`（scale / valid_key）被 msopgen 忽略
  （生成时告警 `The "attr" value is invalid or no "attr" exists in the map`）；
- ⇒ scale / valid_key 暂时通过 **kernel 内编译期常量 + 后续 tiling** 传递，
  或改用支持的 attr 描述格式（S1a-1 时再解决，不影响当前正确性验证）。

## 四、状态

| 阶段 | 状态 |
|---|---|
| S0 工具链 | ✅ |
| S0.5 aclnn 调用链 | ✅（R294） |
| **S1a-0 identity kernel** | ✅ **本轮** |
| S1a-1 q@kᵀ matmul | ⬜ 下一步（cube：LoadData/Mmad/L0C） |
| S1 full FA | ⬜ |
| S2 流水 ≤105 ms | ⬜ |

## 五、下一步（S1a-1）

在 kernel 里实现 `S = q(64,128) @ k(512,128)ᵀ`（FP32 累加、写回 half）：

1. `DataCopy` q/k GM→L1；
2. `LoadData`/`SetFmatrix` 处理 ND→NZ；
3. `Mmad` 到 L0C；`DataCopy` L0C→UB（FP32）；
4. `Cast` 到 half → GM；
5. 扩展 `test_fa294.cpp`：k 用 512×128、输出 64×512，与 CPU 参考对比 `max_abs_diff`/`relRMS`。

通过后即进入完整 FA（softmax 向量段 + PV + 循环），再进 S2 流水。
