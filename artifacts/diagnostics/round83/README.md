# Round83 — resumed FP16 kernel implementation

> **Later shape correction:** [Round84 official checkpoint verification](<../round84/actual_shape_receipt_fixed84.json>)
> confirms MLP intermediate size25600 and TP8 K3200,not the K3456 assumed in
> this round's planning. The preserved K3456 measurements remain valid generic
> GEMM experiments,but are not this model's actual MLP shape. The corrected
> [real TP8 A/B experiment](<../round84/README.md>) ran all50 layers and reports
> its remaining precision and latency gaps without accepting the whole graph.

The direct human “继续吧” request rearmedthepersisted goal atrevision16.
The [preflight](<preflight83.json>) verified the unchangedCPUFP16reference,
immutable proven runtime library and absenceof prior model/NPUworkers.
Onlytheowned910Ahost isused;thetwo unusedCUDAgrantswerereleasedbeforecompute.

## Two disjoint fresh-fork implementation scopes

- [Portable bias epilogue](<bias/tik_bias_epilogue83.py>): HalfA/HalfB cubeGEMM,
  FP32L0C→FP32UB,FP32biasadd,andonlyonefinalHalfconversion. It doesnotuse the
  unsupportedTIKhardwarebiasargumentorforceanA2SoC. Biasvaluesaretheoriginal
  HalfparameterspromotedexactlytoFP32 bytheprivateprobe.
- [Blocked FP32-output GEMM](<blocked/blocked_fp32_gemm83.py>): oneFP32L0C
  accumulatorstayslive acrossallKtiles,thenstoresFP32GM. RealH3matrixsizesfit
  boundedL1/L0C/UBtiles;Mpaddingiszero-only. AllDMAunitsandfive-GMargumentorder
  aredeclaredandtested.

Rootreviewedbothimplementationsandwrote [five-GM runtime bridge](<runtime5gm83.py>),
[CPUHalfbias probe](<probe_bias83.py>) and [H3-sized blocked probe](<probe_blocked83.py>).
TheprobeskeepCPUHalfLinearastheactualnumericreference;FP32Linearisonlyan
accumulatordiagnostic. TheyreportweightpacksetupseparatelyandincludeApacking/
outputunpacking inresidentkernelend-to-endtimings.No microtimingiscalled a
completeencoderresult,norissuccessinferredfromsourceorcompilation.

## Actual two-K-block smoke result

The [smoke receipt](<results/blocked01/blocked_smoke_result83.json>) hascompiled,
registeredandexecuted M128/K256/N128. Thisexercise**two**KblockswithoneFP32L0C:

- repeatedFP32outputsarebit-identical;
- relativeRMSversusFP32accumulatordiagnostic2.880e-7;
- finalHalfoutputversusactualCPUHalfLinear9.891e-6,sameasuredasnativeHalf;
- packedkernel0.207ms,residentwithlayouts0.538ms,nativeHalf0.141ms;
- this ismicrocorrectnessandexplicitoverhead,**notanH3speedup**.

The [portable bias result](<results/bias01/bias_result83.json>) thencompiledand
executedtheFP32UBbiasaddwithonefinalHalfcast. Againstthe**sameCPUHalfLinear**:
relativeRMS3.366e-6versusnativebiasedHalf3.518e-4(about104×lowererror),with
bit-identicalrepeatsandunchangedCPU/GPUoperands.Thisisa microprimitive result,
notfullencoderprecisionacceptance.

FullH3KVandQprojections completedinthefirstblockedrun,withFP32accumulator
diagnosticrelativeRMSabout1.32e-6. Thelargerup-projectionattempttimedoutbefore
itsresultreceiptwaswritten;it isnotclaimedasmeasured. The
[timeout disposition](<static_timeout_diagnosis83.json>) preservesexit124andthe
[exact worker check](<static_timeout_exact_state83.json>) confirmsalloldworkers
exited. A [compacttile-loop candidate](<blocked_dynamic_steps83.py>) changesonly
theouterPython-unrolledstepstofor_range;KaccumulationandHalfparameter values
areunchanged. Itcompletedexit0inaseparatenewwrite-onceroot,notreplayingtheoldrun.
The [blockedbiascandidate](<blocked_bias_epilogue83.py>) likewisecompletedboth
thesmokeandactualH3-sizedchecksdescribedbelow.

## Actual H3-sized results after compact-loop correction

The [verified summary](<summary83.json>) nowincludesallfourH3projectionshapes.
FP32outputerrorversustheFP32accumulatordiagnosticis1.08–1.32e-6,alloutputs
repeatbit-identically,andzeroMpadrows remainzero. ThecompactQkernelproduces
**theexactsameFP32outputhash**astheearlierstatic-loopQkernel. Reducingthe
compilerIRbyusingadynamicoutertileloopdidnotchangethemathematicalresult.

ResidenttimesincludingApacking/outputunpackingwithcachedHalfweights are:
KV0.605ms,Q0.763ms,up1.430ms,row1.291ms. NativeHalfLinearcontrols are0.225ms,
0.367ms,0.662ms,0.721msrespectively. ThesecustomkernelsretainFP32partials
whilenativereturnsHalf,sotheyaresupportingprecisionwork,**notaspeedupclaim**.
ForunbiasedQ/up/row,thefinalHalfcast'sCPUHalferroristhesameasnative;KVbecomes
repeat-stablebuthasslightlygreaterfinalHalfCPUerror. Bothfindingsarepreserved.

The [H3-sized blockedbiasresult](<results/blocked_bias01/blocked_bias_kv_result83.json>)
alsoexecuted M2043/K5120/N128. AgainstthesameCPUHalfLinear,theone-castFP32UB
biaspathhas1.354e-5relativeRMS versusnativebiasedHalf3.523e-4(about26×lower).
RepeatedoutputsandallCPU/GPUoperandsareunchanged. Its0.245mspackedkernel-only
timeexcludeslayoutworkandisnotreportedasthewholeencoder'slatency.

[Finalverification](<final_verification83.json>) confirms23productionfiles,13API
files,CPUgoldreceiptandruntimelibraryunchanged,andnoremainingworkers.Theowned
910Aleasewasreleasedafterallrunsended. Allrequiredfresh-forkresultswerecollected.

## Acceptance and remaining work

AllmodelweightsandCPUFP16gold remainunchanged. TheR82same-conditiontextresidual
2.48%andnativecompleteerror7.42%remainunresolveduntilactualkernelintegration
andfullgraphcomparison. Fullencoder<2s,co-residentliveencoder/DiTserviceandofficial
A2/A3task-matchedwholepipelinenumbersremainactivegoals.Noafter-datanumeric
thresholdorBF16cross-devicegateisintroduced.
