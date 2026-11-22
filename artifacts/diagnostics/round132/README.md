# R132 — audio residual first-change diagnostics

[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) remainsInProgress;
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) staysBlocked. [Continuation scope](approval132.json)
is diagnostic only: no accepted newbaseline, tolerance, copycandidate or49-stepmedia.

## Recorded findings

[Verified R132 findings](RESULTS.md): same8workers completed9full50forwards/rank,
11,160strictchecks/3,600FAlaunches/0errors. Firsttailchangeinrank0/1 isfinalAdaLNoutput,
whileaudioheadinputalreadychanges391F32words. Other6ranksdidnotchangeinthesetwo
observedpasses. OriginalaudioheadfixedFULLinput10replays/rank isbit-identical,
so thisrun doesnotattributethefirstdelta toFP32audio_out orTPgather.

[Cleanup](cleanup132.json) confirms allprivateprocessesexited/Falseflagsrestored/key
removed. TheleasewasretainedonlyforthefollowingR133originalsmallcomponent,then
releasedafter [its cleanup](../round133/cleanup133.json). NeitherTruebaseline nor
copycandidate/precisiontolerance isadopted; #15staysopen/#4Blocked.

[Worker RPC](diagnostic_rpc132.py) keeps actualTP2+USP4, original50blocks/155checks,
FP16matrices/originalFP32exceptions and nativecopy/offload. Planned9fullforwards/rank:
False1 → Truecontrol3 → Truetailtrace2 → original localheadreplays10 → Truerestored2 → False1.
The local10calls areseparate, with0extra155checks orTPgather. Noencoder/VAE/generate.

## Tail observations and exact local proof

[Trace](audio_tail_trace132.py) follows sp_gather/time/finalnorm/AdaLN/finallayer,
video/audio localdispatch-beforeTP and headpostTPgather. FULLlogicalSHA andselected414
audio-row SHAareseparate; video inferenceoutputstabledoesnotprove allhiddenstable.
SHAonlylocatesaboundary; actualaudioheadFULLinput/weight/bias andpre/postTPoutputs
arecompared over **everyFP32word** using [bounded chunks](paired_words132.py).

The modelalwayscomputestheFULL41792×5376headinput andlocal41792×16audiooutput;
414rowsareselectedonlyformeasurement andthenby theunchangedoriginalforward.
Frozen inputisindependentbeforeoriginalapply; originalresult isfrozenbeforeTPgather,
neverreplaced. SharedmethodinstancesroutebyactualLayeridentityandrestoretheirprior
descriptor/override state. Constructorfailure rollsbackpartialinstalledobservers.

OnlyoneCPUfullinputreference andoneNPUfrozeninputareheld (~0.837GiB/rank each),
plus smalloutputreferences. Devicefree/MemAvailableguardsfailclosedratherthan
subsettingcompute. Fixedinput/actualparameters/biasarematched tooriginalCPUref
word-by-word beforelocal replays. Original/frozenlayoutsareseparate; physicalformat
is explicitlyunmeasured, notassumedidentical. Dispatchobject/sourceSHAisnotCANNkernelattribution.

## Nonclean controls and safety

ImmediateD2H/SHAchanges scheduling; sameflagTruecontrol andobserver-restoredTrue
controlsarepreserved. False1isnotnative-self-repeat proof. Allnumericdifference
metrics areseparatefromflag/hook restoration and arenot error tolerances.

[Portablecontracts](test_contracts132.py) and [small CPUunit](test_unit132.py) validate
sourcecontrol/wholewordmetrics andpartialconstructorrollback, notNPUmodelprecision.
[Fresh inputpreflight](preflight132.json) rechecks8on-hostfixtures. [Launcher](launch_diagnostics132.py)
usesnewprivate root/session/0600key,verifiedPIDfdcleanupandmonotonic90minute budget;
noSDKchange,device reset,sharedprocesssignal orCI/dormant. [Collector](collect_diagnostics132.py)
transfers onlyboundedJSON/text; weights/inputs/activationsstayonthehost.

Exactsourcepins,actualreceipts andnumericfindingsarekeptseparately afterexecution.
Ifheadinputalreadydiffers, do notattributepropagatedoutputs tolocalGEMM. Onlya
same-FULLinput/weight/bias local-repeat failure supportsalocaldispatchfinding;
postTP-onlychangesrequiretheir owncollective/selection investigation.
