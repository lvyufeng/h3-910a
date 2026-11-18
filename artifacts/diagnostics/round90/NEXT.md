# Next — real TP8 full encoder, useful vision fixes not external conditioning

## Verified R90 facts

[Actual summary](<summary90.json>) confirms NPUrequestedGELUapproximateNoneandtanh
produceidenticalbitoutputson3realHalfinputs,whereCPUerfandtanharedistinct.
ExplicitTorcherfFloatbetterbut7e-5;newactualFloat-vectorerfGELUcutsmergererrors
4.33e-4/3.10e-4→3.76e-7/7.88e-8,withouttouching27MLPtanhmodules. This semantic
fixismuchcloser,butnotperfectbitexactandnotawholemodelprecisionpass.

OriginalFloatvec_reluunsupportedcompilefailureispreserved.Newsource
[erf_gelu_fixed90](<erf_gelu_fixed90.py>)5a5e85...replaceguardwith0.5*(x+abs(x)),
legitimateFloatUB math,noA2SoC. Actual2024Bkernelobjectc0406299...;
8192smoke1864Bobject197f8529...;onefinalHalfcast.
Only4mergerinstancesperModuleType,actual32kernelcalls/6fullvisioncasesproved.
Negativezero+0vsCPU-0differencepreservedinrawextremecontrols,numericallyzero
butnotdeclaredbitidentical.

Whole27visionjointpatch+linearbias+TIKerferror0.7243819%,resident1.366966s,
onlyaslightnumericalimprovementfrompatch+bias0.7246477%,1.364952s. Actualinputs,
weights,oldCPUgold/HF/libraryfrozen,allnative/repeatedoutputsrecovered.54tests.
Lease217andfreshunused133/134grantsreleased.Nohardwarejobsremain.NewNPUrunneeds
newownedclaim;noCI/dormantorotheragentsleases.Allnewagentsfreshforks(notresumeerf90-fresh).

## Next real complete encoder run

PrepareRound91fromR87bestnative-rowencoderdriverasaread-onlysource,noteditit.
1. Original351HalfvisionweightsfromR78verifiedconversion; samefixture8064patches/
   images48×84×2,2043texttokens,actual50rawlayersbeforefinalnorm.
2. InstallR87onecastvisionbiasadapter8393... +R89patchadapter2eb5... +correctedR90
   4mergererfadapter5a5e85...; allseparate switchesandoriginaldefaultdisabled.
   Compile/cachesoutsidewarm,pins/checkalloriginalHalfparams. NoFloatlinearweight
   shadows,R84rowFP32adapternotinstalled;correctMLP3200local andKVpadR81unchanged.
3. Prepareallkernelsseriallybefore8rank model loading;copyverifiedobjects/caches
   toprivateRound91dirwithresolvedpathchecking,notcompile/raceinallranks. R90
   erfbuildercachehitsreturnmetas;primechecksoriginalR79visiondtypeHalf.
4. Fullcase matrix at least(nativevision+nativeattention),
   (correctedvision+nativeattention),(correctedvision+R85preciseattention/nativerows).
   AlsoCPUinitialtextconditioncontrolto isolate residualtext arithmetic,never
   use its time asfullencoderwarm. All50rawlayers/all3DeepStack vsCPUHalf tensors,
   rankequalityandrepeats/native-recovery/parameterpixelsSHA gates.
5. Actualtimingmustcallstockprivate`encode_ids` includingREALnewvision,50text
   layers,HCCL/attentionpackingandD2H. Warm3or5runsmaxof8rankspercallreportmedian,
   alloutputbitsmatch thatcase'suntimed fullactualgraph output. Donot timeCPU
   precomputedconditionorgoldentry;don'thidepacking/GC/layout/runtime-sync overhead.

BestR87precisetext/native-rowfullencode2.898743s withnativevision7.5915%gap;
correctedvisionpathmayormaynotimprove fulltextprecision—R84/R85 proved thatmore
accurateleavescanamplifydifferently. NativeattentionR82baseline2.180sneedsanother
~0.2sreduction,butexactunder2srequiresactualsamecasewarmgatesnotextrapolation.
R90vision~0.1sgainonlypromising,notalreadyunderspeedtarget.

## Remaining precision/speed choices, without endless blind Float patches

Ifwholeprecisionstill~percent,compareCPUinitialtextcondition2.4757%native/
1.8193%preciseattention once withsameweightsagainstpriorR85SHA. BracketrawCPU
midlayerinputerrorswithR86resetnotfalsefulltiming. Do not combineFloatrowagain
withoutnewreason(preciseattention+F32row2.7148%,3.064sslower). WholeCPUHalf
linearbitorderdifferences1e-5canamplify;checkfullraw-boundarydriftandhighrange
tokensnotjustonebenignleaf. LayerNorm/TanhFloatwideningR89nobenefit;don'trepeat.

Forrealvisioncost,head72fusedstreamedFAcanreplaceNPUtorchFloatQK/PVandR88'sslow
24-launch/imageprobe,butoriginalscale72/noCausal/cropkey4032andCPUHalfMATHleaf
proofneedpreserve. ExistingR88frozenQKVusesprovenance3fb5... nottextR85capture
491452... becausehead72/16differsfromtext128/8. No BF16force ornewtolerance.

Liveencoder+DiTco-residencyandmatchedA2/A3fullpipelinecomparisonstillunfinished.
R76DiT49mean14.657/VAE13.5retainscope;notend-to-endnewFP16conditioningvalidated.
Goalrevision16active/armed.Newownedleasebeforework,allnewsubagentsfreshforks.
