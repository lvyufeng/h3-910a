# R302 — 分块 Cast 未解决 AICore 异常；下一步锁定「用户工作区接口」

## 一、本轮两次尝试

1. **方案 A：让 Matmul 直接输出 FP16**（`cType = ...half`）
   → 编译失败：`kernel_operator_mm_base_impl.h:662: no matching function for call to 'BroadCastVecToMMCal'`
   ⇒ 该 Matmul 配置不支持 FP16 输出（至少当前模板参数组合不行）；
2. **方案 B：分块 Cast**（C 保持 FP32，4 次 16×512 的 `DataCopy→Cast→DataCopy`，UB 峰值 32 KB + 16 KB）
   → 编译/安装成功，但**仍然 `ACL_FAIL 507015`（AICore 异常）**。

## 二、结论：UB 超限不是主因

方案 B 把手工缓冲从 192 KB 降到 48 KB，崩溃依旧 ⇒ 问题在 Matmul 框架本身的工作区使用上。

## 三、最可能的根因

`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mm)` 里的
`GetSysWorkSpacePtr()` 返回的是**框架维护的系统工作区指针**，它与 `workspace` 参数的关系
（是否等于 `workspace` 起始）需要确认。我当前假设它等于 `workspace+0`，于是把用户 C 缓冲放在
`workspace + tilingData.sysWs`。若框架实际把系统工作区放在别处（或按 `GetUserWorkspace(workspace)`
返回用户区），则：

- 我们把 C 写到的地方可能仍在系统工作区内（互相踩踏）→ AICore 异常；
- 或者 `sysWs` 值与框架实际保留量不同 → 越界。

## 四、下一步（R303，优先级明确）

1. 在 AscendC 头里查 `GetSysWorkSpacePtr` / `GetUserWorkspace` 的定义与语义
   （`asc/include/adv_api/matmul/*.h`、`aarch64-linux/include/ascendc/**`），确认：
   - 系统工作区是否由框架在 `workspace` 内偏移分配；
   - 用户缓冲应该用哪个指针（`GetUserWorkspace(workspace)` 还是 `workspace + sysSize`）；
2. 按正确语义改 kernel 的 C 基址；
3. 若仍崩溃，用「最小形状 16×16×16 + 单位矩阵」跑，排除 tiling 形状因素；
4. 目标：`relRMS ≤ 1e-3`。

## 五、状态

- 交付不变：**16.34 s/step（−8.9%）**，49 步 ×2 验证，relRMS 在原生带内，`H3_PERF_COMBO=1` 默认 OFF；
- AscendC：S0 ✅、S0.5 ✅、S1a-0 ✅；S1a-1 编译 ✅、**运行期 AICore 异常（工作区语义）**；
- 机器 ts-217（910A）租约至 20:52 UTC。
