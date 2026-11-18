# Next — integrate FP32 partials selectively,then verify the full FP16 graph

## Verified R83 primitives

1. PortableFP32UBbiasaddfollowedbyoneHalfcastnowactuallycompiledandran.
   MicroM128/K512/N128: CPUHalfrelativeRMS3.366e-6 vsnative3.518e-4(104×lower).
   ActualH3-sizeM2043/K5120/N128:1.354e-5 vs3.523e-4(26×lower). SameCPUHalfgold,
   sameoriginalHalfparameters,bit-stablerepeats,noFP32weightshadows.
2. UnbiasedHalfA/HalfB→FP32GM GEMMsranforallfourH3sizes:KV,Q,MLPup,androw
   (M2043,K3456,N5120). FP32diagnosticrelativeRMS1.08–1.32e-6,repeatedoutputs
   bit-identical,paddedMrowszero. TheactualFP32outputiswhatfutureHCCLmustcarry.
3. GenericM128/K256/N128smokeverifiedtwoKblocks.FP32L0CstaysliveacrossallK;
   thereisnoHalfpartialandnoFP32UBpartialsum.
4. LargePython-unrolledoutertileloopsledtoacompilertimeout. ThecompactTIKouter
   for_rangecompiledin5.5–7.5sanditsQoutputmatchesstaticQbit-for-bit.Kblocks
   remainstaticallyinitializedfirstTrue/thenFalse.

## Exact sources and their scope

Use [compact FP32 builder](<blocked_dynamic_steps83.py>) and
[verified five-GM runtime](<runtime5gm83.py>) forlargeprojections. The
[one-cast blockedbiasbuilder](<blocked_bias_epilogue83.py>) isverifiedatM2043K5120N128.
Itusesonlythe[reviewed blockedplan source](<blocked/blocked_fp32_gemm83.py>),pinned
byhash;thecase-specificbiasproofisin[theactualreceipt](<results/blocked_bias01/blocked_bias_kv_result83.json>).
Do noteditanyexecutedsourceinplaceorblindlyreplayexistingwrite-onceroots.

## Next real-model precision experiment

Patchonlytheisolatedcandidate'srow-parallellinears(o_projandMLPdown_proj) to:

- cacheHalfweightsinK1N16 order,verifyingprefix/rawbits and sourcepointer+offset+
  version (differentviews sharingstoragearenotmutations);
- packtheoriginalHalfactivationswithzeroMpadding;
- launchtheFP32-outputkernel;keepitsresultFP32throughHCCLsum;
- onefinalHalfcastafterthecollective,notbefore;
- preserveoriginalparameterhashesandrestorethenativepathforanA/Brecoverycheck.

FirstmeasureoneactualtextlayeragainstsameCPUFP16inputandCPUgold;thenthefull50
layerswithexactCPUFP16conditionsandnativevisionconditions,bothrepeat-verified.
UseR82'ssame-condition2.48%gapandnative7.42%gapassavedcontrols,notnewtolerances.
TheunbiasednativeKVpathisstillslightlyclosertofinalCPUHalfonasinglecall,but
non-repeatable;thecustomKVpathisstable.Don'tclaimindividualHalfaccuracygains
wherethemeasurementdoesn'tshowthem.

Theactualo_projTPshapealsohasK1024/N5120;buildandverifyitbeforefullgraphuse.
TheMLPdownshapeK3456/N5120isalreadymeasured. Wholeencoderclockmustinclude
activationpacking,outputunpacking,andtheFP32collective;cachedHalfweightsare
extraNPUmemoryandmustbeaccountedforintheco-residentservicebudget.

## Bias integration in vision

UseoneFP32biasepilogueaftercubeGEMM andonefinalHalfcast,nevertheunsupported
TIKhardwarebiasarg. CurrentblockedkernelsrequireN/Ktilemultiples;visionMLP
sizesmayneedadditionalzero-onlyK/Npaddingandcropcontrols. Verifyrealvision
weights/inputsandall27layerboundariesbeforedeploying.NofabricatedFP32teachers,
nooldBF16outputcasts,noafter-dataacceptancethreshold.

## Performance qualifications

R83residentwithlayoutsoverheads:KV0.605ms,Q0.763ms,up1.430ms,row1.291ms;
nativeHalf0.225/0.367/0.662/0.721ms. CustomsretainFP32,nativereturnsHalf.This
isnotaspeedupyet;FP32partialprecisionmustimprovetherealmodelbeforepayingthe
extraoverhead. Fuse/persistpackingonlyafteraccuracyisverified.

Wholeencoderlastmeasured2.1805s(R82),not<2s. DiT/VAEearliermeasurementsretain
alltheirqualifications;liveencoder/DiTco-residencyandofficialA2/A3task-matched
wholepipelinecomparisonremainunfinished.

AllR83workersended,23production+13API sources/CPUgold/launcherverifiedunchanged,
andownedts217leasewasreleased. LiveclaimownedagainbeforeNPUwork;releaseunused
CUDAgrantsimmediately,noCI/dormantpoolopt-in.Allnewsubagentsmustbefreshforks,
neverresumeinactiveoldones.Goalrevision16isarmedafterthedirecthumancontinue.
