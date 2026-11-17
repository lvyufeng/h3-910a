# Next: preserve FP32 GEMM partials,then add a portable bias epilogue

## Verified starting point

- Sameindependentofficial-HF CPUFP16 oracle fromRound78,allhashesunchanged.
- Round82 real50-layerTP8 conditioninginterventions: native7.4201%relativeRMS,
  CPUinitialonly4.3257%,CPUDeepStackonly7.1941%,bothCPUconditions2.4757%.
  Everycase repeatsandrestoringnativeconditionrecoversRound81'sexactoutputhash.
  Theseare nonlinear interventions,notadditiveerror attributionornewtolerances.
- Nativecompleteencode_ids warmedrank-maxmedian2.180496s;stillabove2s.
- A new **FP16×FP16→FP32-output TIK kernel actually compiled and ran** for
  M128/K512/N128. RepeatedFP32outputsbit-identical;relativeRMS4.081e-7 against
  FP32accumulatordiagnostic. Castingitsoutputonce toHalf has1.3247e-5relativeRMS
  againsttheactualCPUHalfLinear,identicalto thenativeunbiasedHalfcontrol.
- Thisismicrocorrectness/capability evidence,notH3sizeperformance or fullencoder
  precisionacceptance. NoBF16agreementisrequired.

## Exact implementation to build on

Use the [executed TIK5GM builder](<gemm/tik_gemm_legacy5_82.py>) and
[executed probe](<gemm/run_tik_legacy_probe82.py>),not the generic DSL scheduling
prototype. The5GMABI reusesthealreadyprovenruntime [libraryreceipt](<gemm/runtime06/probe_fp32_out.json>).
ExtraauxinputsareunusedHalf32Bbuffers—notmathchanges. InputsareK1M16/K1N16,
outputN1M16,withthehostreversepermutationvalidated. Thebinary reallyisAiCore
andtheFP32outputpreserves lowbits beforeanyHalfnarrowing.

All8physicalchipslive-report30cores. SDKAscend910Bprofilealsohas30;the
32-coreAscend910Adefault isnottheruntimecapacity. Neverconfuse32GiBmemorywith
32cores. ActualshortSoC is **Ascend910**,despitethefull/runtime910Blabel.

## Bias correction required

TheTIKhardware/L1biasargumentwas**rejected beforecompilation/launch**:
`matmul param bias,not support Ascend910`. Do notfakeanA2SoCorforcepastthis
hardware/APIguard. TheSDK-onlyclaimthatfull910Blabelwouldenablebiaswasnottrue
onthispart.

Implementinstead:
1. unbiasedHalfA/HalfB cubeGEMM ->FP32L0C;
2. FP32L0C ->FP32UB (alreadyexecuted);
3. addtheoriginalHalfbiasvalue,promotedtoFP32,inUB;
4. oneexplicitFP32->Halfconversion at theveryend.
This matchesCPUHalfLinear'sbiasroundingorderwithoutFP32weightshadows. First
microtestagainstsameCPUHalfLinear,andagainstactualnativebiasedHalfLinear;keep
alloriginalweights/inputsunchanged. Thenextendto blockedH3 shapes M2043/K5120
and row-parallelN5120,using30coresandboundingtailreads/writes. Includealllayout
conversioncosts in thewholeencodermeasurement.

ForTP rowprojections,keepcustomFP32partials throughFP32HCCL reduction andonly
then narrowonce. A Float32collective afteraHalfpartialcannotrecoverlostbits.

## Avoid repeated setup dead ends

GenericTBE DSLattempts02/03/04neverlaunchedakernel: missingCCEtarget,missing
operationbankcontext,thenemptydynamiccomputecontext. Theirreceiptsandsources
arepreserved. DirectTIKBuildCCE bypassedthoseschedulerproblemsandsucceeded.
Do notblindlyreplayanyexistingwrite-onceroot or changeexecutedsources.

## Resources and goal policy

Round82workersfinishedandtheownedts217leasewasreleased. Liveclaimownedagain
beforenewNPUcompute;thearbiterselectsCUDAhostsfirst,soimmediatelyreleaseany
unusedCUDAgrantsbeforework. NoCI/dormantpoolopt-in.Allnewagentsmustbefreshforks;
neverresumeinactiveoldagents.

Wholegoalremainsactive: fullCPUFP16numericagreement,completeencode<2s,live
encoder/DiTco-residency andofficialA2/A3task-matchedcomparisonarenotcomplete.
