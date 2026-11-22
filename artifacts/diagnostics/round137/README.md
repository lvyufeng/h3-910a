# R137 — unchanged original fixed-input component repeatability

[Human approval](approval137.json) allows fixed-input encoder andfixed-native-latent
audio decoder diagnosis withoriginalmath/defaultFalse only. No candidate, True policy,
newtolerance, DiTforward,49sampler orvideoVAEcompute. [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15)
continuesInProgress; #4copyremainsoff.

[Component RPC](component_rpc137.py) prepares presentation/keyframes once viaoriginal
pipeline.encode_prompt, capturesactualnativeencoder._encode ids/visionkwargs, freezes
word-identicalNPUinputs andrestorestheoriginalmethod. Replays useoriginal
pipeline._encode_text_hidden, retainingencode_ids inference/cudnn SDP/staging semantics,
actualTP8 andnativeHalfencoder. Nooldhidden/teacherfeatures areusedasencoderinput.

Audio usesR136originalFalse finalaudio latent `[2,32,207]` withsourceSHA andcomplete
CPU-v-fixedNPUwordgate, thenoriginalaudio.decode_latent (includingFP32latentdenormalization),
nativecomponent-residency/acceptedhalfNaNcompatibility scope. ThefullFP32decoderis
replicatedindependentlyoneachrank, **not decoderTP8**. Nativeclampisnotreplaced.

Eachcomponenthasnative8 → trace2 → observer-restored4 calls/rank; wholeHalf/F32output
comparisons,inputallocation/formatandimmutableCPUmaster/statechecks areseparatefrom
logicalSHAtrace. Tracehashingcanchange schedulingandisnonclean; traceagreementdoesnot
establishkernel determinism orsupersedeuntraced repeats.

[Boundary observer](boundary_trace137.py) returnsNone, hashesimmediateinputs/outputs,
never retainsTensoraliases anddeferscaptureerrorsuntilfullcomponentend. Original
BigVGAN uses**parametrizations.weight_norm**, so noextra`.weight`propertyread or
parametrize.cached()isallowed. Actualderivedweight isseenonlyinnativeParametrizationList/
weightNormchild forwardPOST. Currentaudio moduletrace isregion-level resblocks/up/
pre/post, notallAMPconvolution/activationinternals. SHA isnotphysicalpadding/CANNproof.

[Minimal bootstrap](integration_hooks127.py) excludesR13649/requestsavewrappers and
refusesDiTforward. [Controller](launch_components137.py) embedsprivatecapabilityinchildenv
atPopen, andindependentlycleans verifiedprivateprocessesafterpublish/thread/pidfdsetup
failure. [Driver](generate_components137.py) markscompleteonlyafterworkandengineclose;
metadata/closeprimarypreservationisbest-effort, notanIOguarantee. ExactPIDfd UID/env/
startticks and90min totaldeadline/parentexitchildrenreconcile remain; noreset/sharedjobs.

[23 observer mocks](test_boundary_trace137.py), [controller fault mocks](test_controller137.py)
and [portablecontracts](test_contracts137.py) checksafety only, notNPUmath/support/repeatability.
Originalsource/weights/inputs/outputsarchives remainhost; onlysource/JSON/textiscollected.
Actualcounts,timing/allocator/numericfacts arepublishedafterexecution,allprivatehooks/
methods/IPC/processesrestored andownedlease releasedbeforecompletion.
