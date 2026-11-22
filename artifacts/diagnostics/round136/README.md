# R136 — approved original49 quality comparison, not adoption

[Approval](approval136.json) authorizes three complete requests: original defaultFalse,
temporaryglobalTrue＋finalAdaLN candidate, restoredFalse. This tests the **combination**:
the global flag can affect live encoder/VAE as well as DiT. It does not isolate one
operator's causal quality effect or prove full49 candidate repeatability (one candidate request).
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) remainsopen; no tolerance/adoption/copy.

[Driver](generate_quality136.py) retains theoriginal50blocks,49actualevaluations/50sigma
boundaries,TP2+USP4/encoderTP8/VAEpatch8,FP16matrices andoriginalFP32protections.
[Integration](integration_hooks127.py) keeps155Halfchecks and measuresactualFA/candidate
hits everyfullstep. [Control](quality_rpc136.py) prepares/all8sets/finishes native→candidate→
restored modes, verifiesCPUmasters/residentstate andallowsa physical-onlyabortrestore.

[Candidate](fp32_bias_candidate136.py) keepsR134/135HalfGEMM→FP32output/bias→oneHalf
arithmetic, admitting originaltwo time rows onstep0andthree ontheother48steps. No padding,
subset,parameterrebind,clipping,BF16orapproxcache. Runtime support for2rows isnotproved
byMock; unsupportedrequestsfailclosed, noHalfmm-upcastfallback.

[Native extraction](native_media136.py) returnsarrayswithoutfloatcasts, codecdecoding or
pixel/audioedits. `nativeframes136.npy` uint8 RGB and`audio136.npy` FP32stereo, final
latents andown-liveencoderhidden arehostonly. CRF18/yuv420p/AAC MP4isapreview, **not**
a rawqualityreference. Encoderconditioning is recomputed, never a teacher.

[Qualitymetrics](quality_metrics136.py) comparealllogicalHalf/FP32words andFP64bounded
errors,rawRGBpixels/PSNR/perframe,andrawstereoaudioRMS/SNR/correlation withnogain,
alignmentsearch,resampling ornormalization. [Runner](compare_quality136.py) rejects
incomplete modelteardown/changedsource/fileSHA, thenonlyCPU-loads sixnativePTfiles
andmmapstheNPYfiles. Fullarrays remainhost; onlyscalarJSON returned. Original/restored
noiseandPSNRarenotautomaticcandidate tolerances.

[Mock](test_adapter136.py), [portabletests](test_contracts136.py) and [CPUmetricsunit](test_quality_metrics136.py)
checkcontracts/math butarenotNPUexecution/qualityproof. [Launcher](launch_quality136.py)
usesnewprivate root/key,90min totaldeadline,atomicfatal andexactPIDfd-boundcleanup
includingdetachedchildrenafterparentexit. Neverreplayunknownmodelwork,resetdevice,
changeSDK orsignalsharedjobs. Exceptionmetadata preservingprimaryisbest-effort,
notguaranteedunderarbitraryIO/closefailure; everyfailureisreported.

Actualsourcepins/receipts/timings/qualityfindings are recordedafterexecution. Endall
privateprocesses/IPC,restoremethods/Falseflags andreleasetheownedlease. Anyadoption,
precisionthreshold,copyorotherapproximationstillrequiresa newexplicituserdecision.
