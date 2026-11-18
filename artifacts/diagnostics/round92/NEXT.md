# Next after actual same-input block25 isolation

## What is now established

[Summary92](<summary92.json>):newvision/native-text raw24error0.253075%,through
originalCPUHalfblock25error0.269741% (1.065857×). OnthesameHalfinput,NPU25
arithmeticRMSnative7.77246e-5/precise7.31962e-5/FloatrowOnly6.66017e-5/joint
5.24251e-5. Noneaccountsforthe17.04%R91completeoutputerror.EveryoriginalCPU
block25gateSHA a83... matchesR78 twice;originalSDPA maskNone/is_causalTrue;
noFloat/BFteacher/finalnorm/DeepStack. All8repeats/native recovery/input/weight
SHAs andR91/R86continuationhashesverified.30tests pass,oldgold/sourceunchanged.

Actualfrozeninputsare [pinned](<executed_sources92.json>) at
`/tmp/h3_round92_case02/NPU_raw24_inputs92.pt` (fileSHA2aa10844...,62,813,067B).
Correctedvision/native-text raw24tensorSHAe232f61b4260b81b08bb0cf3eb96a7df6d0a56ed3c0c82ab9d0ec0f15004a528.
CPU original Half25outputs filef3b6cd...83,733,652B;NPU4×4outputfile44c94f...334,731,175B.
Donoteditoldexecutedsources CPU838027/prefix668ad7/NPU66f794/wrapperf53e0c.

FailedfirstCLITypeError optionaltensor_infoNone shadowing publishednoresult;
[record](<results/cpu01/CPU_block25_failure92.json>) retained.NewCPUroot02wrapper
passesreviewedmetrics,no mathematicalcorrectionorgatewaiver.

## Round93 — CPU-only late propagation, now preparing/running

Freshfork`cpu-suffix93-fresh`/task42writesonlyRound93/cpu. Rootmustcollect/review
its source beforeexec and neverresumeoldinactiveagents.

- Newroot `/tmp/h3_round93_suffix01` with immutableCPUoracle838027... staged.
- OnlyoriginalCPUHalfdecoder25..49,25modulesretained(~24GB),16CPUthreadsMKLDNN,
  original64-headSDPBackend.MATH,HFM-RoPE/maskfromR78originalinputs.NoDeepStack
  afterraw24,nofinalnorm,noFloatlinearweights/orBFgoldoutputcasts.
- BaselineoriginalCPUraw24→25..49 mustreproduce**all25R78rawboundarySHAs** before
  readingchangedactualNPUstate.Thebaselineisoriginalsource+originalweights,
  notanewteacher orweakenedreference.
- Thenactualcorrectedraw24e232→originalCPU25..49 twice.Requireall25repeatSHAs,
  finiteoutputs,unchangedparams/input/gold/HF/source. RecordeveryboundaryRMSvs
  verifiedoriginalbaselineandthefirsterrorjump;finalselected50 pre-norm remains
  thesameH3consumerboundary. Thisisinput-onlysensitivity,notencoderwarm.
- ComparethisCPUlatepropagationtoR91nativeNPUpropagation. IfCPUalsoamplifies,
  upstreamvisionprecisionmust improve; ifnot,locatetheNPUlatearithmeticfirst
  diverginglayerusingthenewCPUsame-inputblockcontrolstyle. Do not assumeeither.

R92NPUworkersarezero;owned217leaseheldforCPUcontroluntil20:21:10Asia/Shanghai
(`2026-11-18T12:21:10+00:00`). AllNPUcomputehasended. IfCPUworkwilloutlivelease,
coordinatewithleasearbiterbeforeexpiry;do not silentlyreuseanexpiredclaim or
modifyothers'leases.NewgoalroundcontinuationmustinspectRound93STATEandjobs.

AftertheactualCPUjobfinishes,collectits receipt/progress/failure/log,runrelevant
tests,hashsource/gold/weights/payloads,verifyallworkersendedthenrelease217.
NoCI/dormantused;freshunusedCUDA133134released. Beforea finalreportcollectall
stillrelevantjobs andrequiredfreshforks. Goal16active;nooverallcompletion/block.

## Independent speed route

R91puregeometrycache preservesall77boundarybitsandnativefullwarm2.2776→2.2074s.
NoR92newfullspeedcase.Neverreplaceactualvision+50textwarmwithCPUprecomputed
conditioning. Astaticinput/positionmaskcachewouldrequirevalueguardsforunversioned
inference tensors plusfull77/outputproofsandnew5paired8rankmaxwarm samples.
BadR91correctedvision/fulltextcombinationremainsnotdefaultregardlessofbetter
leaves/earlylayers. OfficialA2/A3same-task/precisionmatchedbenchmarkunfinished.
