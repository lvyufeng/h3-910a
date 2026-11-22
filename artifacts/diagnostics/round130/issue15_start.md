<!-- h3-910a:issue15:R130-start -->

## R130继续：原time proj_out保护路径的四GEMM/FP32累加拆分

#15仍由lead／lvyufeng执行、Project In Progress，无blocked_by；#4仍Blocked。
已读最新Issue/Project与R129记录，当前commit56ec8f5已提交。R129实测firstchange在
sameinput/weights的time proj_out local-output、TPreduce之前；内部rootcause仍Unknown。

### 本轮窄范围

- 只加载原FP32time权重和R128ownlive NPU timesteps，用trueTP2+USP4/world8。
- 原ResidualFP32Projection123.apply保持原4Half-GEMM/FP32out、三scaledadd及bias顺序，
  scale4096/4096²、FP32parameter、高低残差值不改。
- 首先 original→instrumented→restored；scoped observer只在原apply中记录每GEMM的
  actualoperand/output，以及每inplaceadd前后值的snapshot，返回原result，不把副本作计算结果。
- 固定同一GEMM operands/alpha累加inputs做rawrepeat：判断首个变化是GEMM输出、scaleadd
  或异步存储复用／写回路径。任何observer/D2H改变调度都单列，不当clean性能或修复证明。
- 逐wordcount/maxabs/relRMS证明时注明全范围；只SHA时明确fingerprint，不混称完整叶子gate。
- 输入/FP32params/缓存high-lowweight words与format/stride/alias保存，不启copy/pool/NUMA。

### 资源/精度/预算

只ownedliveclaim ts217，extraGPU立即release；租约至12:53+08。
复用on-hostcheckpoint/fixture不本地下载或中转。sharedSDK/driver/CI/dormant不改。
仅小组件，无新49生成；完整model155/FP32例外/TP基线不修改。预算≤60分钟后review。

若找到最小修复，也须独立exactrepeat/full50门，不因为“多sync变稳定”就默认合并。
任何可测数值policy变更停用写数字由用户决定，native漂移不当容差。
source/input/weightsSHA、失败/恢复与actualgroup/phase由本Issue持续记录，结尾commit/cleanup/release。
