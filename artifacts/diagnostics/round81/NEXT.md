## Round81 actual result and next controlled experiment

The KV-only zero-row padding now has full-model evidence,not just a small
operator repeat:

- all8 ranks completed the real27-layer vision and50-layer text path;
- all8 first/second hidden states were bit-identical;
- 3warmed calls used the private candidate'sactual `encode_ids`,with proof hooks
  removed,and included vision/DeepStack/text/final `.cpu()`;
- same-call rank maxima were2.234800s,2.155799s,2.213749s;
- warmed median2.213749s is **not yet below2s**;
- the extraKVcache is262,144,000B perrank,allFP16,100entries with unchanged
  prefixes and zero tails;
- embedding numericerror is exactlyzero;the one signzero bit difference was
  independently explained;
- complete selected50 CPUFP16 relativeRMS remains7.4201%,so this is **not a
  precision acceptance pass**.

The small text-only subset has0.0739% relativeRMS,but visualtokens have7.8914%.
The2033 reference-low-range rows have10.5391% relativeRMS.This rules out describing
the error as solely high-range/sink outliers. The subsets arediagnostics,not
newtolerancesorclampingpolicy. EveryCPUreference value remains unchanged.

### Next step: isolate vision conditioning from text arithmetic

A newdiagnostic should keep the same50-layer TP8FP16 textweights/math/KVpadding,
but feed exactly the alreadyverifiedCPUFP16 inputs_embeds and threeDeepStack
inputs to the realNPUtextmodel. Compare its selected50 against the sameCPUgold,
then restore nativevisionconditionand require the originalnativeoutputhash to
return. This can identify whether correctingvision arithmetic will remove most
of the7.42%fullgap,before investingin a newFP16operand→FP32accumulator bias/TP
kernel. It is *causal input isolation*,not a production encoder optimization or
external-conditioning speed claim. Keep native fullencode2.214s as theactual
reference,and do not substitute the isolationtime for wholeencoderperformance.

### Kernel directions after isolation

1. NativeHalfbiasedLinearmeasuredextraHalfbiasrounding;an AscendC/TBEFP16operand
   FP32accumulator→FP32bias→singleHalfepilogue removes that identifiedrounding
   difference withoutallFP32weightcaches.
2. TProwpartialoutput shouldstayFP32 throughcollective andonefinalHalfcast;
   nativeHalfinputs/F32groupedGEMMfailed910A,andfloatcollective alonecannotrecover
   Halfpartialprecisionlost.
3. Probabilitycompensation reducesarbitraryF32PVerroron910A,butPythonproxycost
   alreadyexceeded2svisiononly. Fuseitifactuallyneeded;do notdeploytheproxyas
   successfulperformanceoptimization.
4. Co-residentliveencoder/DiTserviceandofficialT2VA vsFL2VAcomparabilityremain
   unproven.NoBF16crossdevicegateisreintroduced.

Allnewagentsmustbefreshforks;inactiveoldagentsmustnotberesumed.Liveclaimowned910A
againbeforeNPUwork,becausetheRound81leasewasreleasedafteralljobsended.
