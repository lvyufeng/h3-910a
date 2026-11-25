# R294 — 决定性修正：算子必须按 **Ascend910A** 生成，aclnn 调用链在 910A 上已打通

## 一、发现的真问题：SoC 版本不匹配

首次调用 aclnn 自定义算子失败：

```
GetWorkspaceSize rc=161001 msg=... Failed to execute operator AddCustom_0.
Reason: 1. SoC version ascend910 verification failed. ...
```

- `161001` = `ACLNN_ERR_PARAM_NULLPTR`（`acl/…/op_errno.h:15`），但**真实原因**是 SoC 校验；
- 我们一直用 `-c ai_core-Ascend910B4` 生成，`op_host` 里是 `AddConfig("ascend910b")`；
- 而本机运行时报告的 SoC 是 **`ascend910`（910A 家族）**，平台配置里同时存在
  `Ascend910A.ini` 与 `Ascend910B1/B2C/…`；
- 大部分 910A 机器**不支持** AscendC？——**实测否定**：`msopgen gen -c ai_core-Ascend910A`
  生成成功，`AddConfig("ascend910")`，`bash build.sh` **编译打包成功**。

⇒ **AscendC 在本 910A 上可用**，此前失败纯粹是 SoC 选错。

## 二、aclnn 调用链已打通（S0.5）

用 AddCustom 模板包（kernel 仍是 TODO 空实现）验证整条链路：

```bash
msopgen gen -i add.json -c ai_core-Ascend910A -lan cpp -out proj
cd proj && bash build.sh
bash build_out/custom_opp_ubuntu_aarch64.run --quiet --install-path=/tmp/h3_r221/oppA
source /tmp/h3_r221/oppA/vendors/customize/bin/set_env.bash
```

C++ 宿主程序（`test_add293.cpp`，约 90 行，已入库于 /tmp/h3_r221）：
`aclInit → aclrtSetDevice → aclrtMalloc/Memcpy → aclCreateTensor → aclnnXxxGetWorkspaceSize →
aclnnXxx(ws, size, executor, stream) → sync → 回读对比`。

实测输出：

```
tensors tx=0x… ty=0x… tz=0x…
workspace=0
RESULT n=1024 maxdiff=1.8877 bad=1021 first=(-0.3230,0.2991)->0.0005
```

输出全 0 是**预期**：模板 kernel 是 `// TODO: user kernel impl`，没有计算。
重点是 **op 执行成功、无报错、显存与 buffer 通路正确**。

## 三、编译/链接配方（已固化）

```bash
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
source <install>/vendors/customize/bin/set_env.bash
CANN=/home/mseco/Ascend/cann-9.2.0/aarch64-linux
OPP=<install>/vendors/customize
g++ -std=gnu++17 -O2 -o test test.cpp \
  -I$OPP/op_api/include -I$CANN/include \
  -L$OPP/op_api/lib -lcust_opapi \
  -L$CANN/lib64 -lascendcl -lnnopbase -lpthread \
  -Wl,-rpath,$OPP/op_api/lib -Wl,-rpath,$CANN/lib64
```

AscendC 编译还需（R292 已记录）：

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/aarch64-linux-gnu:/usr/include
export CPATH=$CPLUS_INCLUDE_PATH
```

## 四、对 S1 的影响（重要）

1. **重生成 FA 算子必须用 `-c ai_core-Ascend910A`**（此前 `/tmp/h3_r221/faop` 全部作废）；
2. S1a 的实现与验收方式不变：写 kernel → build → install → 用同一套 C++ 宿主对比 torch；
3. 910A 的 AscendC 算子可用性已经实测，不再是假设。

## 五、下一步

1. `msopgen gen -c ai_core-Ascend910A` 重新生成 FA 算子；
2. 写 S1a（q@kᵀ 单块）kernel，build + install；
3. 扩展 C++ 宿主为 FA 测试（q/k 随机填充、与 CPU 参考对比 `max_abs_diff`/`relRMS`）；
4. 通过后扩展完整 FA → S2 流水（目标 ≤105 ms/块）。
