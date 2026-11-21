<!-- h3-910a:followup:native-cpu-half-packing-segv127 -->

主路线图：#1；发现于#3的R127b正式engine初始化。
初始优先级P1、组件Runtime、状态Backlog；本Issue是未解决风险，不是新精度许可。

## 已执行事实与未知项

- R127a的8rank首请求原50块×49完成；R127b在请求开始前报rank5退出−11。
- 原始C++栈在`at::native::DEFAULT::direct_copy_kernel(c10::Half)`／`copy_`，
  发生在加载完模型后的layerwise offloader初始化附近。
- 没有原始Python tensor/line，**无法单凭该日志确认具体_to_cpu行、越界、别名或UAF根因**。
- R127c仅增加CPU packing元数据/span检查和Python faulthandler；其初始化及三个请求均成功。
  这不是根因修复证明，也不代表该崩溃永远消失；重现频率未知。
- 六个contiguous／transpose／offset的小CPU synthetic pack组合原始位相同，
  不代表真实50块／大权重布局的初始化可靠性已验证。

## 目标

在不改原FP16／FP32数学、pinning／stream策略或共享SDK的前提下，得到有界最小重现／
具体失败阶段、真实storage ownership和恢复策略；把可靠性与性能、精度门分开。

## 诊断清单

- [ ] 复用owned已核验主机权重，固定torch／torch_npu／CANN／driver／source SHA；
      不本地下载／中转weights或activation，也不占用dormant／CI。
- [ ] 先setup-only／small native packing重现，不为收日志反复运行完整49步。
- [ ] 记录target对象id、storage base pointer／nbytes、logical pointer、offset、shape／stride／dtype、
      dtype累计pack offset／total span；只记录scalar metadata，不改变copy／分配／unbind顺序。
- [ ] 区分跨块同一个Parameter对象、不同Tensor共享storage、合法offset view和非法span；
      同data_pointer不是自动别名漏洞，metadata合法也不是nativecopy无崩保证。
- [ ] 保留failed begin/no-end、Python/C++stack与环境；不把marker等待／传输问题归因为模型精度。
- [ ] 禁止驱动reset、改共享CANN／TorchNPU、换BF16/HF32／截断权重来绕过；
      任何实现改变必须有同输入完整wordproof与fail-closed回滚。

## 验收／退出

- [ ] 实际失败点／恢复条件由执行证据支撑；无法重现如实写Unknown，不宣称修好。
- [ ] 若修复，至少覆盖实际尺寸／strides／aliases／循环offload/reload，原参数和全155checks保持。
- [ ] 若涉及数值改变，停用并转Awaiting approval；只读诊断不是精度晋升。
- [ ] 写回Issue＋Project、关联commit／PR；private进程／IPC清理与owned lease释放。

## 现有证据

- [R127开发与已执行失败来源](https://github.com/lvyufeng/h3-910a/commit/6d84b97)
- [R127b失败摘要](https://github.com/lvyufeng/h3-910a/blob/6d84b97/artifacts/diagnostics/round127/failed_lifecycle127b_summary.json)
- [CPU packing诊断代码](https://github.com/lvyufeng/h3-910a/blob/6d84b97/artifacts/diagnostics/round127/init_diagnostics127.py)
- [small native synthetic unit](https://github.com/lvyufeng/h3-910a/blob/6d84b97/artifacts/diagnostics/round127/init_packing_unit127.json)

本Issue保持未完成，不把R127c退出0当作native CPUcopy可靠性修复。不作为近似／有损加速的许可。
