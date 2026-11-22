# R131 — user-authorized full50 deterministic diagnostic only

[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) remainsInProgress;
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) staysBlocked.
[User approval](approval131.json) authorizes diagnostics, **not** adoption of a new
precision baseline, tolerance or copy-stream candidate.

## Recorded outcome

[Full findings](RESULTS.md) and [verified actual receipts](verified131.json) show the
same8workers completed all9forwards/rank,11,160checks and3,600FAlaunches without
wrappererrors. Deterministictime **andvideo** predictions arebit-repeatable, but
**audio still differs**: maxabs6.0796738e-6/relRMS0.0000343518%. No full-output bitgate
or decodedmedia qualitypass isclaimed. Native/deterministic/restored diagnostics
warmtimes are19.927/20.013/19.905s in theexactdeclaredscope.

[Cleanup](cleanup131.json) confirms allprivateprocessesexited andflagsFalse restored,
keyremoved; ownedlease released. Keep thisdiagnosis,Reject automaticadoption ofthe
still-nonrepeatable fulloutput baseline. #15 staysInProgress and#4Blocked.

## Execution contract

[Private worker RPC](diagnostic_rpc131.py) calls original`VerifiedDiT.forward` under
originalcontexts, with all155Halfchecks,50blocks,TP2+USP4 and originalFP32protections.
The planned cases are nativeFalse3 → deterministicTrue3 → restoredFalse3, on the same
own-live step25 inputallocation. These are **prediction tensors returned by one
full50forward**, not49-step sampled/decodedvideo orVAEquality results.

ActualTP/SP/Ulyssesgroup sizes,local28heads and FAwrapperBHNDhead7 shapes arechecked;
50originalFA launches/pass mustbeobserved. No guess from theconfigurationalone is
reported asan executedproof. [Driver](generate_diagnostics131.py) verifies eight
same-worker receipts,allcases and11,160strictchecks beforecalling the experimentdone.

The timeboundary observer takesone same-current-stream clone,neverchanges the
originalreturn anddoesnot readD2H untilthewholeforwardhasfinished. Fulloutputs/CPU
comparisonreferencesstayonthehost andneverparticipateinsampling. Inputvalue/layout
fingerprints andoriginalCPUmasters(residentparams/buffers) arechecked; masters also
match theoriginalR129reference, notjust a newbefore/afterselfcomparison.

Timing iswhole50forward plusoriginalaudits/outputD2H andthetinytimeclone, **not clean
production orpure-kernel timing**. For coordinatedpassordinal, computeactualmaxrank
andthenmedianwarm repeats separately. Same-case repeatability,candidatevsnativefirst,
candidatevsfrozennativewarm-last,andphysicalflagrestore aredistinct fields.

## Bounded private control

[Launcher](launch_diagnostics131.py) usesa newexactprivate root/session,0600authkey,
atomicallypinnedreceiptandno uncertainlaunch replay. [Cleanup](private_abort131.py)
opensPIDfds,rechecksUID/environment/startticks andsignals theboundtask; it doesnot
signal reusablePIDnumbers. Parentexit/nonzero/fatal/RPCtimeout allreconcile detached
privatechildren; a totalmonotonic90-minutecompute deadline doesnotresetonfileevents.
Boundedteardowngraceisadditional; unknownstateisreportedratherthanclaimingcleanup.
No sharedprocesssignals/SDKchanges/driver/device resets orCI/dormant resources.

[Portablecontracts](test_contracts131.py) testsource/approval/ref/control boundaries,
notactualmodelprecision. The [CPU-only pidfd unit](test_private_abort131.py) exercises
ownedtoychildren and a separate-capability survivor; noTorch/NPU/model is involved.
[Collector](collect_diagnostics131.py) transfersonly boundedJSON/text,neverweights,
inputs oractivations. Numericalfalsegates areordinaryresults, notfailedexceptions.

## Preserved prelaunch corrections

[Initial preflight failure](preflight_failure131a.json) records systemPython3.10
missing`hashlib.file_digest`; no modelhadbeenlaunched. Compatibility streamingSHA
reverified all8fixtures in [fresh preflight](preflight131.json). The actualworker
interpreter remains theverifiedPython3.12whereR129usedfile_digest successfully.

Readonlyreview caught PID-number-signal andparent-onlycleanup limitations, plusan
incorrect3-D FAshape assumption; allwerecorrected **beforemodellaunch**. The actual
FAentry accepts rank4BHND, soH isaxis1. Nonewas silently described as a successful
runtimeexperiment. The exact launchedsource pins andobservedreceipts arepreserved
separately once launched; R130source/resultsremainimmutable.
