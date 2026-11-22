# R135 — approved full50 final-AdaLN candidate diagnostic

[Approval](approval135.json) authorizes full50 **diagnostics only**, notnumericadoption,
tolerance,49-step sampling/media orcopyoptimization. [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)
remainsInProgress, [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) staysBlocked.

## Recorded full50 outcome

[Results](RESULTS.md) and [verified summary](summary135a.json) showall8candidate
video/audio predictions repeat exactly for3full50calls/rank. Original/restored
video repeats exactly butaudio retains15FP32word drift. Candidate changes video
3,578,877/3,773,952FP32words (relRMS0.007187915%) andaudio13,245/13,248
(relRMS0.008823783%) versusoriginal-underTrue. ThisisnotdefaultFalsequalityproof.

Actual11,160checks/3,600FAlaunches/0errors/24candidatePythoncalls verified;
warmcoordinatedrankmaxmedian20.201/20.179/20.177s andPyTorchpeak7.13499GiB.
[Cleanup](cleanup135.json) recordsFalse/warnFalse andmethodsrestored,privateIPC/
processescleaned,ownedlease released. No candidate adoption/newtolerance, no49media.

[RPC](diagnostic_rpc135.py) uses theoriginal50graph/155Halfchecks on sameown-liveinput,
actualTP2+USP4/FAheads andoriginalFP32protections. Original3 → candidate3 → restored3
areallunder **the same temporaryTrue** condition; outputreferenceisnotoriginalFalse.
Initial/finalFalseflag restoration andmethoddescriptor restoration areindependent
from numericalagreement. Input/CPUmasters/residentstate/oldreferenceSHA arechecked.

[Adapter](candidate_adapter135.py) routes onlyactual`final_layer.adaln_proj.linear`;
sharedmethod'sunknownlayersstrictlydelegateoriginalboundargs/return. SameHalfmatrices,
FP32GEMMoutput/FP32bias→oneHalfreturn aretheR134candidatearithmetic. Roundingschange
andaremeasured,notaccepted. Actualexpectedcandidatecalls1/candidateforward=3/rank,
24aggregatearePythonpathcounts,notCANNkernelattribution. Noothermodules/weightschange.

Observerspreserveoriginalreturns;timehasoneconsumerstreamclone,onlyreadD2Hafterwhole
forward. Normalandexceptionpathsattemptallindependentadapter/flags/time/FAcleanup,
recordfailures/unknownstatewithbest-effortprimaryerrorpreservation,notaguarantee
underreportIO failure. Constructorinstallationis
transactionalandsuccessfulownrankhandleisheldevenifanothercohortpeerfails.

[Portabletests](test_contracts135.py) and [puremock](test_adapter135.py) checkcontrol,
shared-methodtargetroute/kwargs/descriptorrestoration,notactualNPUprecision. [Launcher](launch_diagnostics135.py)
hasnewexactprivate root/session/auth0600key,PIDfd+UID/env/startticks,atomicfatal/
90-minutecompute deadline andboundedteardowngrace. NoSDKchange/device reset/shared
processsignals/CI/dormant. [Collector](collect_diagnostics135.py) onlytransfersbounded
JSON/text; weights/inputs/activationsstayonthehost. Actualsourcepins/receipts/results
arepreservedafterexecution, no uncertainmodelreplay orautomaticnumerictolerance.
