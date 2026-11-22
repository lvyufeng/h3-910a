## R133原小组件诊断合同（#15延续）

R132trace首次变化为rank0/1的final_layer.adaln_proj输出SHA，音频头输入391F32词先变；
原audio_out固定FULL输入10次/all8均位同，不能指认audiohead或gather根因。
继续只原time+finalAdaLN组件，不采用True为新baseline，不放宽误差或启动copy候选。

- 复用host原weights/own-live timesteps。FP32time params+原四HalfGEMM保护，
  原finalAdaLN Halfweight/input/bias、SiLU、ColumnTP2gather保持，USP4/world8。
- 首先再核fixtureSHA、原AdaLNparameters对R132residentSHA；原time生成的t_emb必须匹配
  R132正式worker真实finalAdaLN输入SHA/shape/stride后才能进入重复。
- 用原worker create_diffusion_vllm_config factory，构造前set_current_vllm_config。
  installeddefault Pythondispatch callable+sourceSHA对R132已测audiohead匹配；
  R132具体finalAdaLN VllmConfig未独立记录，CANN内核更未知，不冒称全部context/物理格式相同。
- 32次原Module（含TPgather）以observer立即冻结localinput/output、保留原return；
  撤销apply wrapper后128次原boundlocalapply固定Halfinput/bias，最后4Module复核。
  不换成torch.mm/F.linear、FP32运算或减行，不忽略bias，不early-exit某rank。
- Half输出用uint16全词比较，FP32time为uint32。坐标及每个最大差异记录，不当容差。
- 全8cohort gate在setup/ref和每Module/局部边界；exception原fatal原子回执触发parent监控。
  20min总deadline、pidfd+exactprivateenv/UID/starttick处理failure及detachedchildren，无device reset。
- #15仍InProgress／#4Blocked。lease为ownedts217续持到15:56+08，只此组件，无CI/dormant。
  结束False/warnFalse恢复，私有process退出、release lease。源码先commit/push/pin，
  tests和只读review不是NPUproof；真实rawJSON/误差再记录。
- 没有full50/155新检查、没有49generation/encoder/VAE/decodedmedia质量声明，
  inputs/weights/activations均留host，只源码/JSON/text返回client。
