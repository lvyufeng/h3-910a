# Round93 — original CPU Half alone reproduces the late precision cliff

[Actual receipt](<results/suffix01/CPU_suffix_reference93.json>) has completed:
25originalHFCPUHalfdecoderlayers(25..49),275originalconvertedHalfparameters,
16CPUthreads,SDPBackend.MATH,originalrotary/mask. NoFloatLinearteacher,noBFgold
outputcast, noDeepStackafterraw24, noextra finalnorm.

## Reference gate

OriginalCPUraw24→25..49reproduces**all25R78rawboundarySHAs exactly**,including
selected50raw49`82944620...`,beforeanychangedNPUinputisread. Itmatches the
original [R78reference](<../round78/README.md>),notanewnorm-or-Float-weightbaseline.
Source reviewcaughtandcorrected an author'sserializedfileSHA98302...mistakenfor
tensorSHA829446...beforeanyCPUexecution. Theactualtestsnowcheckbothmetadata
names ratherthancopyingtheliterals. No numerical gate was weakened.

## Actual input-only propagation

Theactualcorrectedvision/native-text NPUraw24capturee232f61... has0.253075%RMS
vsCPUraw24. Propagatingthat**same frozen input onoriginalCPUHalf** twice gives:

| Original raw layer | CPU-only propagated RMS vs original baseline | Increment |
|---|---:|---:|
| 25 | 0.269741% | 1.066× over input |
| 40 | 0.865580% | 1.135× |
| 41 | 0.960788% | 1.110× |
| 42 | 1.052459% | 1.095× |
| **43** | **16.648505%** | **15.818679×** |
| 49 | **17.161829%** | 1.005× |

Completeinput→selected50amplificationis**67.8133×**;maxabsoluteerror7926.5.
All25changedboundarySHAsmatchbetweenbothfullsuffixrepeats,finite,andallparams,
input/gold/HF/helper/source/payloadSHAsunchanged.

**NoNPUarithmetic is needed to obtain this17%scale.** This demonstrates strong
input sensitivity in the original CPUFP16 model,firstlargejumpatlayer43; it
cannotidentifywhichlayer43primitive(attention/MLP/etc)isresponsiblewithout
additional same-input instrumentation.Noris17.16%−17.04%avalidadditiveisolation
ofNPUerror,asthelaterCPU/NPUstatesalsodiffer. [R92sameinputblock25](<../round92/README.md>)
remainsvalid:thatblock'sNPUarithmeticwasonly5.2–7.8e-5relativeRMS.

Theupstreamvision/inputprecisionnowtakespriorityoverblindlypatchingNPUlayer25,
43,orchangingdtype/normtocoerceapass. ThebadR91fullvisioncombinationstaysnotdefault.
CPUHalfreferenceandallhigh-range/visualoutputs are retained; no tolerance waiver.

## Independent primitive results

[Round94](<../round94/README.md>) auditedtruepositions/frequencies/Halfnormswhile
CPU93wasrunning. NativeFloatrsqrtonthesameCPUvariancetensorisbit-exactCPU;
Newton1/2 slightlyworsen RMSNorm,so no redundantNRkernelwillbeinstalled.Text
HalfcoshasoneULPnative-vsCPUdifferences(7.14e-6relative);aCPU-exact**position-only**
RoPEconstantinterventionisbeingprepared [inRound95](<../round95/STATE.json>),not
animageconditioningcacheorclaimthatthisalonefixes17%.

## Evidence and safety

[Source/payloadprovenance](<executed_sources93.json>) pinsactualCPUdriver6d062f...,
unchangedR92helper838027...,outputs38b7e1...(41.8MB),andactualinput2aa10844....
32source/actualtests pass.[Finalverification](<final_verification93.json>) confirms
23production+13API,oldCPUgold/HF/launcherunchanged,andzero93/94workers. Theowned
217leasewascarriedforwardforthecontrolledNPU95A/Bandthenreleasedbefore20:21
Asiaexpiry;nootherleasesmodified,noCI/dormant.AllCPU93jobsfinishedandcollected.

[Next](<NEXT.md>):R95scopedCPU-exactRoPEtableA/B;thenfastCPU-faithfulvisionattention
ratherthanblindfullmixtures. Wholegoal16remainsactive:CPUFP16agreement,text<2s,
liveencoder/DiTco-residencyandofficialmatchedend-to-endcomparisonunfinished.
