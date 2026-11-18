# Round94 — actual RoPE and RMSNorm primitive audit; no default patch

These are smallsame-inputleafcontrols withthe [CPU93 late propagation](<../round93/STATE.json>)
concurrent. They are not a full encoder warm benchmark or acceptance result.
NoFloatLinearweightteacher/BFgoldcasts, noCANNbuild orproductionedit.

## Real rotary positions, not guessed dimensions

[RoPE audit](<rope94.json>) recomputestheoriginalHFCPUandprivateR79NPUoperators:
realR78M-RoPEpositions[3,1,2043],actualHalfmetadata;realimagegrid[[1,48,84],
[1,48,84]],visionFloatfrequencyrows[8064,36].Bothlearnedparametercountsarezero.
OriginalCPUinvfreqs andvisionfrequencytablematchcandidatebits;alloutputsrepeat.
The oldR79NPUFloatmatmulfrequencyerror0.0204isalreadysolvedbydirectFloatmultiply;
thisauditdoesnotpretendtodiscoverorfurtherfixit.

| Actual native primitive | CPUrelativeRMS | max_abs |
|---|---:|---:|
| Text cos (Half) | 7.14033e-6 | 0.00048828125 |
| Text sin (Half) | 2.80268e-8 | 0.0000038146973 |
| Vision cos (Float) | 5.26580e-8 | 1.78814e-7 |
| Vision sin (Float) | 4.43311e-8 | 1.19209e-7 |

ThetextcosmaxisoneHalfULP. Theseare*not*bit-exactCPUresults,butaunique17%
wholegraphculprit cannot be inferred from them. OriginalHFCos/Sinfrompositions
alonecanbe a scopedCPU-exactconstantcachewithoutcachingpixels/conditioning;
[theRound95fullA/B](<../round95/README.md>) isnowcompleteandnon-monotonic:the
CPUtablehelpsonecombinationbutworsensothers,sonotdefault.
Itisaprecisionalignmentcache,notnative-bit-losslessliketheR91geometrycache.

## Native rsqrt is exact on the same actual variance

[Actual RMSNorm controls](<norm94.json>) useoriginalCPUHalfHFNorm5120 andsame
actualHalfraw24 inputswithoriginalcheckpointblock25HalfGamma.TwoHalfrounds
(normalisedcastthenHalfgammamultiply) arepreserved.

| Actual input | Native norm RMS | Newton1 RMS | Newton2 RMS |
|---|---:|---:|---:|
| Original CPUraw24 | 6.42506e-6 | 6.75900e-6 | 6.76182e-6 |
| Correctedvision/native-text raw24 | 4.45637e-6 | 4.86662e-6 | 4.84196e-6 |

NativeNPUFloatrsqrt onthe**exactCPUvariancetensor**isbit-exactCPUforbothinputs.
VarianceonownCPU/NPUreductionsdiffers~4.069e-7relative. Newtonrefinementdoesnot
help andslightlyworsensoutputhere. Do not installa redundantNewtonRMSpatch or
attributeR91's17%fullerror toan unmeasuredlow-precisionrsqrtprimitive.
Originalinputs/gammaandallrepeatsareunchanged.

[RoPE source](<rope_probe94.py>) / [norm source](<norm_probe94.py>) areisolated
probes,notproductionpatches. 9source/actualevidencetests pass. All94NPUworkers
finishbeforeCPU93finalresourceverificationandowned217release;nospeedgainclaimed.
Goal16active:completeCPUFP16agreement,text<2s,liveencoder/DiTandtask-matched
A2/A3comparisonremainunfinished. Keepallnegativecasesandtheoriginalgold.
