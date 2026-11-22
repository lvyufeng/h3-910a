# R133 — original final AdaLN small component

[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) continues after
[R132 audio-tail evidence](../round132/RESULTS.md). TemporaryTrue isdiagnostic;
no numericalbaseline/tolerance/copycandidate isadopted.

[Component](component133.py) loads only originaltimeandfinalAdaLN weights, using
actualTP2+USP4/world8. The reconstructedtimeembedding mustmatch theactualR132
formal finalAdaLN inputSHA/shape/stride; finalweight/bias also match R132resident
fingerprints before anymodule repeats. Originalcheckpoint inputfixtureSHA ischecked.

NativeWorkerConfigfactory andset_current_vllm_config areused beforeconstruction.
The installedPythondefault dispatch/sourceSHA matchesR132'sobservedaudiohead,
notan independent recordofR132finalAdaLN'sentire config or aCANNkernel guarantee.
Physicalformat/workspace differencesremainunknown; no newbackend isinstalled.

Planned **32 originalModule calls** (Halfinput/weight/bias, SiLU, TPgather), **128
original boundlocalapply calls** on thefrozenHalfinput, then **4 restoredModulecalls**.
Halfoutput comparisonusesuint16wholewords, notFP32uint32 counts. Inputoutput freezes
are immediate/preTP and never replace originalresults. No50-layer/155check claim,
no49sampler/encoder/VAE ordecodedmedia. Fullsmallmatrix shapesarepreserved.

Everysetup/timeSHA/module/local phase hasanall8cohort gate. [Launcher](launch_component133.py)
monitorsatomicfatal andtotal20-minute deadline; [PIDfdcleanup](private_abort133.py)
usesexactprivate env/UID/starttick,includingchildrenafterparentexit. Originalflags
andmethoddescriptor restoreinfinally. NoSDKchange,device reset,sharedsignals orCI/dormant.

[Portable tests](test_contracts133.py) verify source/contracts, notNPUbitproof;
actualsourcepins/receipts/results arerecordedafterexecution. Originalinputs/weights/
activations remainonhost; onlysource/JSON/textiscollected.
