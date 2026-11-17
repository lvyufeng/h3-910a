# Round82 — exact-CPU conditioning isolation and FP32 GEMM prototype

The [independent official-HF CPU FP16 reference](<../round78/final_CPU_reference_verification78.json>)
is unchanged. No BF16 cross-device agreement requirement is reintroduced.

## Actual isolation result

The [isolation summary](<isolation_summary82.json>) verifies all8 complete
rankreceipts,all50 realtextlayers percase,exactCPUFP16 inputtransfer,repeat
identity,unchangedparameters/inputs,andnativeoutputrecovery.

- Nativecompletehiddenstate: **7.4201%** relativeRMS versusCPUFP16.
- CPU initialcondition only,nativeDeepStack: **4.3257%**.
- Nativeinitialcondition,CPUDeepStack only: **7.1941%**.
- CPU initialcondition plusCPUDeepStack: **2.4757%**.
- WithbothCPUconditions,the2033 low-range reference rows stillhave **1.0179%**
  relativeRMS.So the remainingtext-arithmeticgap isnotonlysinkoutliers.

These are nonlinearinputinterventions,notanadditivevariance attribution or a
newtolerance. All3caseoutputs repeat bit-identically onall8ranks. Restoringnative
conditionrecovers theexact[Round81 baseline](<../round81/summary81.json>) hash.

Onlythe restorednative `encode_ids` path hascompleteencoder timings: same-call
rank-max warmedmedian **2.180496s**,includingrealvision/DeepStack/all50textlayers
andfinalD2H. Noisolated-texttime ispresentedaswholeencoderperformance,andthe
small variationfromRound81's2.214s isnotclaimedasanewoptimization.

## FP16-operand/FP32-accumulator kernel work

A freshforkprepared a [minimalGEMMprototype](<gemm/gemm_fp16_fp32out82.py>) and
[SDKsource receipts](<gemm/sdk_receipts82.json>) showingthatF16→F32 accumulation
andoutput areexpressible even thoughthenativegroupedoprejectsF32output.

Rootreviewcorrected thelaunchABI to useTorch'sactualNPUstream pointer andthe
CANN`ascendcl`/`runtime`libraries. Theprobe nowcomparesagainsttheactualoriginal
CPUHalfLinearoperator,withthebias'soriginalHalfvaluespromotedonlyfor thekernel
epilogue;NumPyFP32 accumulationisjustadiagnostic,notanothermodelteacher.

The [live DCMI check](<gemm/live_core_count82.json>) confirms**30 usablecores** on
all8 original910Bchips.[SDKprofile comparison](<gemm/profile_compare82.json>)
shows`Ascend910B`profile30 vs`Ascend910A`profile32. Theroot'sfirst32-coreguard
wasa falsepositive,notanA2 mismatch;32GiBmemoryisnot32cores.

The first [internal-DSL attempt](<gemm/runtime02/probe_fp32_out.json>) failed
becauseitsauto-schedulehadnoCCEtargetcontext. The
[public-DSL CCE-context attempt](<gemm/runtime03/probe_fp32_out.json>) reachedthe
GEMMtilerbutfailedbecausea standaloneoperationcontextwasmissing. Bothlogs and
sourcehashes arekept,andboth receipts explicitlyreport**no kernel launch**.
Thestaticoperationcontextattemptthenhit anemptydynamiccomputecontext.
The directTIK route instead **compiled a realFP32-output AiCorekernel**. A new
Cshimcouldnotfinddeprecatedruntimeheaders,so thecontrolled5GMABI reusedthe
alreadyprovenruntime library withoutmodifyingit.

[Actual FP32-output execution](<gemm/runtime06/probe_fp32_out.json>) nowshows:
- M128/K512/N128,Halfoperands,FP32L0C andFP32GMoutput;
- twolaunches bit-identical;
- FP32accumulatordiagnosticrelativeRMS **4.081e-7**;
- finalHalfcastversusactualCPUHalfLinearrelativeRMS **1.3247e-5**,the same
  measured errorasthenativeunbiasedHalfcontrol;
- noH3geometryperformanceclaim,andnofullencoderprecisionpass.

The separate [bias variant](<gemm/runtime06/probe_fp32_bias_fp16_out.json>) was
rejectedbeforecompilation/launch: `matmul param bias,not support Ascend910`.
Full/runtime910BlabeldoesnotmaketheTIKshort-SoCbiasgatepass. The nextfixis a
portableFP32UBbiasaddafterunbiasedGEMM,thenoneHalfcast—not anA2SoCoverride.
[Next steps](<NEXT.md>) pin theexecutedTIKsourcesandactualrestrictions.

## Scope

[Isolationdriver](<condition_isolation82.py>) and[actual evidence tests](<test_actual_isolation82.py>)
areprivate diagnostics.No productionencoder,DiT,VAE,API or CPUreference isedited.
Theowned910Aleasewasreleasedafterallmodel/GEMMworkersexited andthe
[finalsource/CPUreference check](<final_verification82.json>) confirmed23production
and13API filesunchanged. [State](<STATE.json>) recordsallcollectedjobs.
FullCPUFP16 agreement,completeencoder<2s,liveencoder/DiTco-residency andofficial
A2/A3 task-matchedcomparison remainactive work.
