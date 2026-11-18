# Next after CPU-only cliff and CPU-exact RoPE full A/B

## Current actual evidence, not guesses

[Summary95](<summary95.json>) has6cases×8ranks×5fullstockwarmcalls. CPU-exact
originalHFHalftextcos/sin cachechangesonlypositionmetadata,1,046,016B/rank,
zero learnedparams/FloatLinear teacher/imageconditioningcache. Its wholegraph
precision is **non-monotonic**:
- nativevision/nativeText7.4201%→7.4895%, clocks2.136741→2.158240s;
- correctedvision/nativeText17.0374%→13.6159%, clocks2.037664→2.062419s;
- correctedvision/preciseText14.8014%→17.1336%, clocks2.669211→2.686949s.
Noneofthe5paired8rankmax samplesisunder2s(bestindividual2.00645s). CPUinitial
conditioning controlsalsoworsenwithCPUtable(native2.4757→3.0516%,precise1.8193→
1.9973%);thosearenotfullwarmclocks. **Do notenabletheCPU-RoPEtableasdefault.**
Itisleafcorrectbutnotageneralwholegraphfix. Allbadcasesareversioned.

[CPU93](<../round93/README.md>) gives the stronger causal isolation: originalCPU
Half25..49fromcorrectedNPUraw24aloneamplifies0.253075%→17.161829%,firstmajor
jump43 from1.052459%→16.648505%(15.8187×). Baselineall25R78SHAsexact,changed
25boundarySHAsrepeat. ThiscontainsnoNPUarithmetic factor,so ablind43NPUkernel
patchcannotbejustifiedbythe17% alone. No uniqueattentionvsMLPfactoridentified.
[R94](<../round94/README.md>) nativeFloatrsqrtonexactCPUvariancesisCPUbit-exact;
NewtonrefinementmakesHalfNormslightlyworse. Do not deploy redundantNRRMS.

Allrequirednewagentswerefreshforks andhavefinished/beenreviewed.35R95tests,
32R93tests,9R94tests pass. Original23production+13API/HF/CPUgold/launcherunchanged,
workerszero,alljobscollectedandowned217releasedbefore20:21Asia. Freshunused
CUDA133/134released;others'leasesuntouched,noCI/dormant. Anynewcompute needs
newownedclaim.LiveCPU43states42/43wereNOTsaved,onlyverifiedSHAs/metricsandfinal49.
Do notpretendthatthosefrozeninputsarealreadyavailable.

## Priority: accurate AND fast vision attention, then complete encoder

ExistingR88precisevisionSDPAprovablyreducesactualQKVleaferror~6×butgeneric
24launch/image/high+lowPV pathisslow. Itwasnotyettestedjointlywithallcurrent
R89patchoneHalfbias/R90trueerfGELU/R87linearoneHalfbias. First acontrolledprecision
probe againstunchangedR78fullvision/DS onactual2image8064tokensshouldtellwhether
thatjointpathmeaningfullyreducesupstreamerror andCPU43cliff. Retainanyslow/
non-monotonicresult andneverlabelthisprobeunder2s.

For speed, a newstreamedselfTIK/AscendCFA must use:
- visionheadcount16,headdim72,originalscale72^-0.5,keylen4032 perimage,NOcausal;
- zero-padD80(or128onlyifproved),last64key/querytailcarefullymasked/cropped;
- FloatonlineLSE/scoreexp/reduction withoutFloatweightteacher;
- HalfP-high +Half((P-high.float())*1024) residual-low PVwithFloataccum,
  singleHalf output: cast probability-only approximation mustnotundoprecision;
- write-oncekernelABIrtDevBinaryRegister/rtFunctionRegister/rtKernelLaunch,
  5GMpointers,currentNPUstream,shortSoCAscend910/30usablecores—notA2fakes;
- firstfrozenR88QKVfileSHA3fb5d52.../actualCPUHalfMATHoutput/inputs unchanged,
  2repeats/actualkernelobjectshas,thenfullvision/DeepStack/encoderCPUcomparison.

ReusethealreadyworkingR85batchCube/R87portableFloatUBbias patterns;avoidstatic
outertaskcompiletimeouts(R83fixedbydynamicouterloop)andunsupportedFloatvec_relu
(R90fixedwithabs/add),donotassumeFloatreduce-max/rsqrtISAwithoutaprobe. No
patchingglobaltorch or productionuntilfullprecision+speedproved. Kernelcommand
cachedsetupcannotbetheonlyspeednumber: fullrealencode mustincludeallpixels,
27vision,50text,HCCL/layout/packing/D2H with5paired8rankmaxwarm samples.

## Optional CPU43 factor audit

Toidentifycliffwithin43,recomputeoriginalCPUHalf25..42 forCPUraw24 andactual
e232correctedraw24,requiretheir93SHAs,beforecpu43hooks. Compareattentionout,
post-attentionresidual/Norm,MLPgate/up/SiLU/product/down onbothsame-inputbaselines.
NoFloatlinearweights/teacher ornewfinalnorm.Diagnosecondition sensitivity,not
proclaimuniqueNPU arithmeticbug.Ifupstreamprecisionimproves,rerunfull50samegold.

## Goal still active

R76DiT49warmmean14.657s/VAE~13.5sretaintheiroriginalscope (externalprecomputed
conditioning). WholeFP16CPUagreement,text<2s,liveencoder/DiTco-residencyand
matchedofficialA2/A3wholepipelinestillunfinished. Do notmarkgoalcomplete/block
forconvenience. Freshforks only—neverresumeinactiveoldagents. Allold executed
sources andtheoriginalCPUFP16oracle remainimmutable;noafterthefacttolerance.
