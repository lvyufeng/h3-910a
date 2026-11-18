# Next — explicit erf GELU, then real encoder integration

## What R89 now proves

[Summary](<summary89.json>) and [raw fixed run](<results/vision02/vision89.json>) show:
- Original HF CPUHalf entry and CPUHalfMATH block0 exactly reproduce R78block0SHA
  with the original351Halfvisionweights,16CPUthreads and originalHFsource.
- NativepositionembeddingsalreadyreproduceCPUbits. NativepatchConv error1.631e-4;
  CPUHalfLinearvsCPUHalfConv2.399e-6; newonecastpatchGEMMvsCPUConv2.458e-6.
- Newreal patch+R87onecastlinears cutsfullvision1.4568%→0.7246% and residenttime
  1.476421s→1.381477s. ExactCPUentry+linearsdiagnostic0.7067% isclose;entryreset
  byitself only1.4165%,so entry/linearrounding interact ratherthanoneuniquebug.
- NormandMLPtanhGELUleaferrors5–7e-6;Floatwideninghasnoleaf/full-outputgain.
  Do not add redundantFloatnorm/GELUpaths. R88slowaccurateattentionstaysdisabled.
- MergererfGELUstillhas3.10e-4/4.33e-4CPUHalfgap.NPUFloatF.gelu+Halfdoesnothelp;
  CPUFloatF.gelu+HalfonlydiffersfromCPUHalfby2.89e-7/1.13e-6.

Originalfailedroot01source/logarepreserved.Inferencecontextwasmissingonlyaround
rootleafrun_controls;fixedroot02adds it andpassesall29tests/all11visioncases.
OldCPUreference/HF/production/librarySHAunchanged,allworkersendedandowned217
wasreleased. NewNPUcompute needsa newownedclaim.Allnewagentsfreshforks—do not
continueinactivevision-leaf89-fresh. NoCI/dormantorothers'CUDAleases.

## Next smallest semantic test — no more caller Float GELU

Use the actualHalfinputsandCPUHalfGELUoutputSHA from [frozenpayloadprovenance](<executed_sources89.json>).
Remote310,953,441B payload is `/tmp/h3_round89_vision02/CPUEntryAndActualLeaves89.pt`,
SHA `97f80014b35abdae2af45fc0fcabc3577d5260b025e299192439012c81a697ef`.

ForbothactualmergerGELUs andtheMLPtanhcontrol:
1. RecomputeCPUHalf `F.gelu(x, approximate='none')` (erf) twice andmatchR89's
   storedCPUoutputSHA,not anoldBFoutputcast. ComputeCPUHalftanhseparately.
2. NativeNPUHalf/FloatF.gelu(None) andnativeNPUtanh on the sameoriginalHalfx;
   comparebothwithCPUerfandCPUtanh beforeclaimingthebackendignoresamode.
3. Test explicitFloatformula `0.5*x*(1+erf(x/sqrt(2)))`→Half onNPU; firstquery
   whethertorch.erfFloatisactuallysupportedandhowaccurateitisonrealinputs.
   CPU explicitformula→Half alsocomparedwithCPUHalfF.gelu—notassumedequal.
4. Ifnativeerfisunavailable/inaccurate, implementFloatvectorerf/GELU in a new
   TIKkernel usingverified910AFloatUB/vexp capabilities. KeeporiginalHalfinput
   andoriginalerfsemantics;neverreplaceexacterfwithtanhorsigmoidsilently.
   Pinnedfloatpolynomialerfcanbe animplementationcandidate,but mustmeasureerror
   againsttheunchangedCPUHalferfonactualdense/high-range/smallnegativevalues.
   Halfoutputonecast;carefulwithnegative-tailcancellationandnegativezero.

Applyonlyto4merger `approximate='none'` modules;leave27MLPtanhunchanged.Use
per-instancegatesoriginalmodeoff/explicitmodeon,repeat/inputsSHA/native restore.
Then complete27vision/3DS A/B withusefulpatch+linearbias,notCPUprecomputedentry.
Do not use a betterleaf toself-declarewholeprecisionornewtol.

## Speed and full-encoder validation after the semantic control

ActualTP8full50FP16encoder mustmeasurethe newrealvision path (patch+bias+useful
exacterf),originalnative rows andfixedKVpadding. Existingbestprecisetextattention
2.898743s isnotunder2s; CPUcondition1.8193% isnotanactualvision-pathspeedresult.
Preserve bothnative/precise-text-attentionswitchesandrepeatall8boundary/output/
paramgates. TheR89visiongainalone doesnotprovecompleteencode<2s orprecision.

Ifvisionattentionremainsdominant, a fusedstreamedvisionFA (head72→zero-pad128,
originalscale72,key4032/no causal mask,Floatonline-softmax/high+residualPV) should
replaceR88's24-launch/imagegenericprobe,notbeclaimedfasterwithoutactualCPUHalf
leaf/whole27tests. RootR88actualQKVpayloadisstillpinned forindependentchecks.

All29source/actualtestssafe.Host/sourcecache isreleased;preserveR89 executed
sourcesandusea new root.Nextliveencoder/DiTco-residencyandmatchedofficialA2/A3
wholepipelinecomparisonremainunfinished.R76DiT49mean14.657s/VAE13.5skeeptheir
originalscopequalifications.Goalrevision16active/armed;noBF16agreementrequirement.
