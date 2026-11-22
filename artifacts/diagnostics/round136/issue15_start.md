## R136只获批完整49步质量对照：原False→组合候选True→恢复False

用户在R135全50预测差异报数后选择“只做完整49步质量对照(Recommended)”。
本轮不授权采用、误差阈值、copy或其他近似。#15 live InProgress/ready/owner lvyufeng，#4Blocked。
起点commit3138304；新source/hash先review/pin/commit再模型执行。ownedts217新lease至20:41+08，extra4090lease立即release。
原weights已验证留host /home/mseco/models/MiniMax-H3/FL2VA，revision0919fb684102ef115126612f3fbb84d1027e7317，无client下载。

- 一个正式8worker Omni，原DiT50层、49实际evaluation／50sigma、每step155Half检查、TP2+USP4；
  原FP16 encoderTP8、videoVAEpatch8、原FP32time/front/final/VAE/norm/residual和power2 guards保留。
  无BF16/clip/HF32/teacher/cache近似或少步、sharedcopy/SDK/拓扑变更。
- 输入固定prompt红球、seed1234、1344×768、120→124帧/24fps、first/last keyframe；
  同原initialnoise/sigma fixtureSHA，自己实时encoder/VAE conditioning，不复用teacherhidden。
- 三完整请求：originalFalse／temporaryglobalTrue+finalAdaLN候选／restoredFalse。
  **True影响whole pipeline(encoder/VAE也可能变化)**，不是单finalAdaLN的独立因果质量证明。
  一次candidate完整请求不等于完整49候选重复性已验证；原／恢复差异仅control，不是自动容差。
- 只targetfinalAdaLN local：sameHalfx/w/bias、FP32 GEMMoutput＋FP32bias→一次Half回写；
  原第一步有2个time rows，其余48步3rows，新shape门允许2/3不padding/subset，算术与R134同。
  若2rows的NPUout_dtype不支持立即failclosed，绝不Halfmm后cast冒充，不自动接受数值。
- all8controlRPC prepare/set/finish，modelstate/原R129CPUmasters引用／source核验；每step实际155/FA50与dtype/runtimeflags。
  计划3×49×155×8=182280checks，DiT FA58800/0errors，候选49/rank392总Pythonhits，非CANN机制保证。
- raw返还video_uint8[124,768,1344,3]及audio_F32[2,165600]在host保存，不改pixels/gain/align；
  finalLatents＋ownencoderhidden也留host。只在modelprocess结束后CPU逐全词/FP64chunk比较三pair。
  native uint8 PSNR与decodedaudio误差分开，不拿encodedPSNR／原生noise自动接受质量。
- 原／候选／恢复wall、cold/warm/每step协调rankmax、encoder/videoVAE/audioVAE分范围、实际copy/FA/collective观测和allocatorpeak报实数。
  观察/audit/D2H/latent存储为nonclean，不冒纯kernel或性能达标。rawNPY/latent不传client。
- 独立私有root/session0700/authkey0600、cohortatomicfatal、90min总deadline/PIDfd+UID/env/starttick、parentexit detachedchildrencleanup。
  失败所有local恢复独立尝试/metadataunknown，保留失败不重跑不抹除；禁止reset/sharedsignal或CI/dormant。
- 先source/Mock/CPU metricunit/readonlyreview/GHpin，再模型；结束所有flags/methods/source/IPC/process清理+lease释放、commit回写#15/#4/#1/Project。
  仍Awaiting approval任何采用/放误差/新近似，不因质量指标较好自动合并。
