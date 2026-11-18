# Round90 — real GELU mode bug and executed Float-vector erf correction

Original [CPU FP16 reference](<../round78/README.md>) andmodelHalfparametersremain
unchanged;thisroundcorrectsthe4mergererfGELUsratherthanreplacing27MLPtanhGELUs.
NoBF16teachercast,Floatweightteacherornewaccuracytolerance.

## Actual backend mode mismatch

[Real preflight](<preflight_summary90.json>) verifiesoriginalCPUHalfoutputSHAs
on3actualfrozenR89Halfinputs. NPUrequestedNoneandtanhGELUare**bitidentical**onall3;
CPUHalfNone(erf)andtanhare**not**bitidentical. OnmergersnativeNPUNoneagreesvery
closelywithCPUtanh(9.45e-7/2.64e-7),notCPUerf(4.33e-4/3.10e-4). This is anactual
mode-semanticdifferenceonthisoriginal910Apath,notaninferredBF16precisionissue.

ExplicitNPUFloat`0.5*x*(1+erf(x/sqrt(2)))`→Halfissupported,buthas7.11e-5/7.71e-5
mergerCPUerfgap.The newlyexecutedFloat-vectorTIK implementationisfarcloser:

| Real frozen merger input | Native None vs CPU Half erf RMS | Explicit Torch erf RMS | New TIK erf RMS |
|---|---:|---:|---:|
| FirstDeepStackmerger | 4.3318e-4 | 7.1073e-5 | **3.7590e-7** |
| Finalmerger | 3.0977e-4 | 7.7119e-5 | **7.8767e-8** |

[Kernel proof](<results/vision02/kernel_probe90.json>) verifies bothactual9,289,728
Halfinputcasesimprove,multiplekernelcallsrepeatbitsandinputsSHAsremainunchanged.
Maxabsoluteerror7.6294e-6.AllrealCPUHalfoutputSHAsreproduceR89;noFloatmodelteacher.

## Full27vision result is still not accepted

[Actual full summary](<summary90.json>) and [raw vision receipt](<results/vision02/vision90.json>)
keepnative/R89patch+linearbiasoutputSHAsunchanged. Sixfullcasesfinished,all27raw
boundariesand3DeepStackoutputsfinite,3warmrepeatsbitidentical,nativerestored.

| Full27 vision mode | Merged relative RMS | Resident median |
|---|---:|---:|
| Native | 1.4568% | 1.470883s |
| Patch+onecastlinearbias | 0.7246% | 1.364952s |
| TIK erf only | 1.4567% | 1.481675s |
| TIK erf + patch+linearbias | **0.7244%** | **1.366966s** |

Theerfsemanticfixworksonactualleavesbutmoveswholemergederroronlyslightly.
CombinedDeepStackRMS0.1481/0.2771/0.4730%;thebiaspathwas0.1536/0.2789/0.4771%.
Fullvision~0.7244% is**notdeclaredprecisionaligned**. TIK addsabout2msinthis
residentcomparison;neitherthesevisiontimingsnorCPUconditiontexttimingsprove
wholeencode<2s.NoCPUprecomputedvisionentryorR88slowattentioninsidewarm.

## Executed hardware contract

[Fixed source](<erf_gelu_fixed90.py>) usesHalf input→FloatUB→A&S7.1.26erfpolynomial,
FloatvexpandreciprocaltwoNewtonrefinements→oneHalfoutput. It is animplementation
approximationcandidate,not a claimofmathematicalbit-exacterf. Sign-freeidentity
`gelu(x)=0.5*x+0.5*abs(x)*erf(abs(x)/sqrt(2))` avoidssigncomparisontensorsand
far-negative-tailspuriousvalues. Scalarcoefficientsareunchangedfromthereviewed
source;actualCPUHalf measurements—notofflineemulation—areabove.

OriginalcompilefailedbeforeexecutionbecauseAscend910doesnotsupportFloatvec_relu.
[Failed log](<results/kernel01/kernel.log>) and [failed receipt](<results/kernel01/kernel_probe90.json>)
arepreserved;newremoteroot02replacesthenonnegativeguardonlywith
`0.5*(x+abs(x))`,usinglegitimateFloatabs/add/muls.NofakeA2SoC/BF16support.
8192smokeandactual9Mgeometrywerecompiledandrun,5-GMABIviaunchangedR83launcher.
The [execution provenance](<executed_sources90.json>) recordsboth2024Bactualand
1864BsmokeobjectSHAs.All32actualadapterlaunchesarecountedseparatelyfrom2268
logicaltiletasks/call;thereviewcorrectedthatbookkeepingbeforeexecution.

Only4original`approximate='none'`mergerinstancesarepatched,original27tanhs
untouched. Signedzeroisnotbitexact:thekernelreturns+0for-0wheretheCPUreturns-0;
[extreme controls](<summary90.json>)preservethatresult,notwaivingithiddenly.
Largepositive/negative65504remainfiniteandnegativetailsarezero.

54tests(41sourcecontracts+8root/fixcontracts+5actualevidence)passed.
[Finalverification](<final_verification90.json>) confirms23production+13API,
CPUgold/vision/selected50/HF,R89frozenpayloadandlauncherunchanged;noworkersremain.
Owned217andfreshunusedCUDAgrantshavebeenreleased.Noothersessionleasesmodified.

[Next](<NEXT.md>):realTP8fullencodercomparisonusingusefulpatch+linearbias+
correctederfvisionpath,withoriginalnativeandprecisetextattentionswitches.
Goalrevision16active/armed:wholeCPUFP16,text<2s,co-residentencoder/DiTandmatched
A2/A3end-to-endcomparisonremainunfinished.
