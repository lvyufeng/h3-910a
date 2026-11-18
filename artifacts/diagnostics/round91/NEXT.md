# Next — isolate late input amplification; no more blind vision/text mixture

## Current verified result

[Full8rank summary](<summary91.json>) has5pairedwarm samples per mode. Noneunder2s:
native2.277611s/cache2.207390s;correctedvisionnativeText2.064467s/cache2.085461s;
correctedvisionpreciseText/cache2.736972s. Geometrycacheisbit-exactonall77boundaries
atall8ranksandhas19,740,672B/rankextra,48realinference-gridvaluechecks(rank0),no
imageconditioningcached. Actualsource682d7f.../geometry10b9f4... and12kernelSHAs
arein [provenance](<executed_sources91.json>).

ThevisionfixesarecorrectatleavesbutwholetextgotWORSE:7.4201%→17.0374%,precise
textonly14.8014%. [Boundarydrift](<layer_drift91.json>) showscorrectedvisionerrors
layer0/9/24roughlyhalfthenlayer49amplifies.NativeCPUinputtextcontrol2.4757%,
precise1.8193% stillmatchesoldR82/R85.NeitherCPUconditioncontroltimeisfullencode.
Do not set the correctedvisioncombinationasdefault or treatthetext-onlysubset's
smallererroraswholeprecisionacceptance. Allbadvisual/high-rangesubsetsretained.

R91peakallocated12,605,280,768B/rankisnotaco-residency proof. Allworkersended,
owned217/freshunusedCUDA133134leasesreleased;newNPUrunrequiresnewownedclaim.
NoCI/dormantorotheragentsleases.Allnewsubagentsfreshforks—nooldgeometry91agent
continuation.40source+actualtests pass;oldCPUgold/HF/production/libraryunchanged.

## Precision priority — a same-input original CPU Half block25 reference

Use a newRound92root with immutableoldR91sources.TheCPU raw24 and25reference
SHAsalreadyexistinR78receiptbutonlyraw24fullpayloadisstored.Gettherecorded
NPUraw24SHA foreachR91casefromtheactualreceipts(notfromnarration).

1. Re-runonlyneededfullprefix0..24 onoriginalHalfTP8weights for(nativevision),
   (correctedvision/nativetext),optionally(precisetext). Controlledforwardhook
   afterrawlayer24savesactualHalfstateandthenstopsall8rankstogether before25.
   RequirecapturesSHA reproduceR91raw24for eachmode;notnewtiming.Thiscapturewas
   NOTpersistedbyR91,so do notpretendanalreadyavailableNPUraw24tensorexists.
2. BuildoriginalHF`Qwen3VLTextDecoderLayer(config,layer_idx=25)`+originalCPUHalf
   weightsonCPUfromofficialcheckpoint,usingR78conversionSHAandHFsourcefcb557...
   OriginalpositionIDs/RoPEfromR78multimodalinputs,originalSDPBackend.MATHcausal
   masksemantics,NODeepStackafterlayer24orfinalnorm. Useoriginal16CPUthreads.
3. OnexactCPUraw24, originalCPUblock25mustreproduceR78raw25SHA, twice. Thatgate
   validatesmask/RoPE/module/load/roundingcontracts beforeusingnewCPUpayloads.
4. Then evaluateoriginalCPUHalfblock25onactualNPUcorrectedraw24, twice. Thisis
   aninput-onlysensitivitycontrol,notFloatweightteacher.PreserveallinputsSHAs.
   SeparatelyrunNPUblock25fromthesamefrozenHalfinput with originalnativeattn/rows,
   preciseattn/nativeRows,FP32row/nativeattn ifnewreason; don'tonlytesttheoldbad
   simultaneousFP32row+preciseattentionmix.
5. Factor errors: CPUFn(NPUinput)vsCPUFn(CPUinput)=inputamplification;
   NPUFn(NPUinput)vsCPUFn(NPUinput)=arithmeticgap. CaptureactualCPU/NPUQKV,
   norm,projection/residual/MLP/rowcollectiveleafinputs/outputsatblock25,
   head-localCPU8of64correspondingtothatrank,selectedqueryprobabilityrows ifneeded.
   No BFteacheroutputcast, noFloatlinearweights/nonewtol. A singleblock CPUHalf
   fulloracleischeapcomparedtorepeatingfull50CPUmodel.

TheR86CPUraw24→native25..49controlhadonly0.5213%gap;thatandR91mid-layerimprovement
supportlateinputsensitivitybutdonotproveauniquehardwareoperatorbug.Theactual
same-inputCPUblock25controldecideswhetherafocusedkernelchangeisjustified.
InspecttheactualH3consumerofrawselected50(normalization/projection)too,butdo
NOTinsertfinalnormmerelytohidebadrawprecision orchangeourboundarycontract.

## Independent speed-only, bit-preserving work

Theoriginal `_get_rope_index` andplaceholder/maskmetadata depend ontokens/grid/
masknotpixelvalues.Theycanbe candidatesforanotherexplicitgatedmetadata cache,
notcachedembedding/imageconditioning. Hash/checkIDs/mmtypes/grid/attentionmask/
configeverycall;unversionedinference tensorsmustvalue-validate. Originaloutputs
computedtwice atprime,all77boundaries+stock/equivalent+cachedSHA proofsagain.
Do not combine timingresultsfromdifferentcases or extrapolateunder2s.

Ifrealvisionattentionremainsdominant,theR88genericprecisionprobe(24launch/image)
isnotafastdefault.AfusedstreamedvisionFA wouldneedoriginalhead72scale/padding,
key4032/unmaskedsoftmax/high+residualPV proofs ontheR88frozenrealQKV.SomeFP32
vectoropsareunsupportedonoriginal910A(Floatvec_relu failedR90);noA2 ISA fakes.

WholeCPUFP16agreement,text<2s,liveencoder/DiTandmatchedA2/A3end-to-endcomparison
stillactive.R76DiT49mean14.657s/VAE13.5skeeptheirscope.NoFloat/BFteacherwaiver.
