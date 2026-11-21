<!-- h3-910a:issue3:R127-start-20261121 -->

## 开始执行：R127 正式engine生命周期事实（lead／lvyufeng）

依赖#2已关闭；Project当前#3为Todo、无执行人。现在认领#3，并转In Progress。
本次只观测原生执行，不把任何新stream／NUMA／数值候选直接投入共享基线。

### 合同与假设

1. 使用已核验原MiniMax-H3 checkpoint（revision0919fb684102ef115126612f3fbb84d1027e7317），
   TP2＋USP4／encoderTP8、8卡，50主块×49实际前向、155strict Half checks／步。
2. 固定prompt red ball、seed1234、1344×768、120→124帧／24fps、同初始noise／50sigma、[0,-1]关键帧，
   每次fresh live NPU encoder＋原FP32 video/audio VAE。不复用hidden／teacher输出、不做近似cache。
3. 在同一个正式Omni worker进程组执行cold请求、第二次warm请求、显式或原生offload后reload请求；
   third请求必须给真实residency／reload证据，不能仅改名冒充reload。先核验正式API与安全生命周期。
4. 观测init、encoder、每步whole-forward、video/audio VAE、load/offload／gather／D2H、
   actual current/copy streams、线程ID、CPU affinity／NUMA与pinned存储状态。
   只在原生边界收集，不每module synchronize；额外详细profile与clean计时明确区分。
5. 默认不启用R126新VAE operand cache／geometry pool／range-selector等候选；
   保留已接受R124＋FA原数学、FP32例外与原NanToNum兼容，来源SHA先核验。
6. source/config/input/kernel fingerprints、设备型号／topology当前事实、所有失败与异常回滚均写入私有R127回执；
   同进程的自然重复差异单列，不当作精度许可。此次不实现Issue#4的runtime优化。

### 资源、预算与安全

已live claim owned ts-217（租约至2026-11-22 00:03+08）；roster旧卡型标注不能代替actual查询。
claim先返回的owned4090 ts-133已立即释放，未在其运行代码。只复用ts-217已有weights，不本地下载／中转。
计划最多3个原49步请求，含init／边界观测总预算约2小时，单request7200s fail-closed上限；
后续具体launch源码与SHA在运行前追加。磁盘仅私有小证据／on-host媒体和必要bounded tensors，
客户端只取source／JSON及小媒体。共享SDK、驱动／设备、原模型目录不改。

### 验收／退出

三种请求同worker实际执行、状态转移／加载／stream／NUMA证据，cold/warm/reload rank-max分别报告；
任何未测项标Unknown。若常驻被32GiB预算限制，报告实际budget与stage策略，不声称已实现常驻。
完成后给#4的具体设计入口；清理私有进程／IPC与release lease。主路线图保持开放。
