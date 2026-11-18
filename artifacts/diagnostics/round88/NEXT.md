# Next — inspect actual vision entry/norm/activation, not a slower PV default

## Proven constraints

[Actual R88 summary](<summary88.json>) shows that more accurate unmasked SDPA on
realHalf[1,16,4032,72] gives leaf1.44–1.58e-5 instead of9.23–9.25e-5, but the
complete27vision gain is only1.4568%→1.1934% jointlywithonecastbias,whilevision
latency1.479→3.047s. Native/R87biasoutputsreproduceoldSHA;allrepeat/native-recovery
andparams/inputs/gold/runtimegatespass. Do not make this slowcandidate thedefault.
Theindependent CPUHalf reference remains valid and no modelprecisionpass was
self-declared. The best complete text path remainsR87native-row2.898743s.

Owned217wasreleasedafterR88worker/sourcechecks;noNPUtaskisrunning. Newhardwarework
requires a newownedclaim,notcachedFREEpermission. Donot touchothers'CUDA/CIleases.
Allnewagentsmustbefreshforks;inactivevision-attn88-freshmustnotbecontinued.

## Next smallest real-leaf protocol

Use a newRound89root and newfrozenfirstnativevisioncaptures,noteditedR88sources.
Reusetheverified originalHalfvisionweightloaderandCPUparameterconversionreceipt.
Capturereal inputs/outputsof:
1. `patch_embed.proj` biasedHalfConv3d: samepreprocessedpixels,channels/temporal/
   patchaxesvalidatedfromconfig,originalHalfweight+bias. CPUHalf**Conv3d** first,
   notjustaFloatLinearteacher. Outputpatchhasonevoxelbecausekernel=stride=patch.
   AlsocompareCPUHalfF.linear(flatpixels,flatHalfweight,Halfbias)withCPUHalfConv3d
   todeterminewhetheraTIKbiasedGEMMreplacementhasanextraorderinggap.
2. Actual block0 `norm1`, `norm2` and firstDeepStackmergerLayerNorm: CPUHalf
   F.layer_norm withsamefrozenHalfinput/gamma/bias,epsandnormalizedaxesunchanged.
   Floatnormalizedintermediatesarepermittedonlytoreproduceoriginalnormsemantics,
   notFloatweightlinearshadows. Include nativeHalfvsFloatthenHalfdiagnostic.
3. Actual block0MLP `GELU(approximate='tanh')` andfirstmerger `GELU()` (erf):
   distinctoriginalmodes,preservethem. SameactualHalfinput CPUHalf output;compare
   nativeHalf andFloatthenHalf. No approximate-activationreplacement/loosertol.

Capture patch projection, interpolatedabsolute positions andfirstblockinput
separately. Reference CPUoriginalHF initialvisioninput canberecomputed cheaply
fromoriginalHalfpatch+positionweights; validate contract (config/inputSHAs), run
CPUtwice, and compare with native. The candidate alreadyusesFloatfour-termposition
interpolationwithoneHalfcast;donotassume thatpositionsarethelargegapwithouttest.
Afull27CPU-initial-inputresetcasecanbracketentryerroramplificationbeforebuilding
anotheroperator. ItmustnotbeusedasfullencodingtimingsinceCPUcomputedinputisnot
insidewarm. Realfullvisiontimingstillincludesactualpatch+position+alllayers.

IfpatchConv double-roundbiasislargeandCPUHalfLinearvsConvagreeswell, addseparate
onecastpatchadapterreusingR87biasbuilder. DeriveactualflattenKfromconfig/weight,
likely1536butcheck first;N1152,M8064needszero-onlypaddingifany. Source pin and
smallmultikernel controlfirst;noA2SoC,biasoperandunsupported,newhardwarefeatures
orFloatweightshadow. Ifnorm/GELUactualerrorsaretiny, don'taddanunnecessaryFloat
patch (R86SiLU showed thatpitfall).

Then onlyapply empiricallyusefulchanges; fullvision27/3DeepStack A/B/nativerecovery
andactualTP8text CPUcomparisonremainmandatory. KeepR88slowattentiondisabled
unlessnewmathematicalevidencedemandsit. Speedpriorityrequiresreducingrealvision
costandtextpacking,not claimingservice<2sfromprecomputedconditioning.

## Remaining objective

FullCPUFP16agreement,text<2s,realencoder/DiTco-residency,andmatchedofficialA2/A3
pipelinecomparisonremainactive. R76DiT49mean14.657s andVAE13.5sretainoriginal
scopequalifications;noBF16agreementrequirementandnoFloatweightteacher.
