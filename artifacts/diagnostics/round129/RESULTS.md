# R129 findings — native repeat drift starts in time projection, before TP reduce

Live [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) remains **In Progress**.
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) staysBlocked; no copy-stream
candidate, precision tolerance or latency claim is promoted.

## Recorded execution, not inferred from an option

- [Original input preflight](input_preflight129.json) verifies the8 prior ownlive
  NPU fixture hashes. Each~44.68MB input remained on-host. Text/video/audio positions
  areunique and disjoint; duplicate-position Half scatter is not supported here.
- [Successful formal-worker RPC result](executed_diagnostics129b/engine_repeat129b/diagnostic_lifecycle129.json)
  has exit0, oneformalengine and same8 workers for threecontrol forwards plus two
  coarseandtwo repeatedtraces: **8,680 original strict Half checks**. These are
  original50-layer diagnostic forwards, **not**49-step sampling or generation.
- [Summary](summary129b.json) verifies inputlogicalvalues/shape/stride/offset before
  andaftereverycall, both reusedallocationandfreshclone. CPUmasters plus materialized
  resident parameters/buffers retained their logicalword/metadatafingerprints.
- Everyforward recorded **50actual FA ACL launches,0wrappererrors**. Noobservedfallback
  switch explains thedelta. This doesnot proveFA/HCCL aredeterministic or rootcause.
- Nativewarmcontrolforward19.926/19.934s versus cold106.856s; traces~40.2s withlarge
  D2H/hash overhead. These scopes are diagnostic andnot optimization performance.

## First changing original graph boundary

Both trace rounds agree on all8 ranks: **`time_embedder` output** is the first
fingerprintdifference. Video/audio/condition projection, tokenrefiner, RoPE and
all earlierobservedboundaries stayed identical. Timeinput itself stayed identical.

The firstregion wasnot aDiTblock, so the second mode named`detailed` in the original
RPC driver repeatedcoarsetracing; it did **not** trace thetimeprojectioninternals.
SHA matches arefingerprints, not complete per-leafdifferentwordcounts. TraceD2H
changesqueue timing, but thedifference persisted intheseinstrumentedruns.

## Narrow original time component — actual TP2, unchanged formula

[Small component script](time_component129.py) loads only originalFP32timeweights
from theverifiedon-host checkpoint. TP2withUSP4/world8 remainsenabled. Theoriginal
R124`ResidualFP32Projection123` guard is unchanged: fourHalfoperandGEMMs withFP32
output/residualsum. This isnot pureFP32GEMM andnot a new numericguard.

[Component summary](time_component_summary129.json) and
[rank0 source/output fingerprints](time_component_rank0129.json) showfourrepetitions:

- Frequency construction andthe firstprojection'slocaloutputandTPgather stayidentical.
- **The firstdifference is local `time_embedder.proj_out` output**, whileitsactual
  inputand originalFP32parameterfingerprints stayidentical.
- Thisdifference exists **before** thatrowparallelprojection'sTPall-reduce. Therefore
  an HCCLreduce isnot needed toproducethis firstlocaldifference. It doesnot ruleout
  other collective differences elsewhere inthefullmodel.
- Finaltimeoutputmaximumdelta is **1.1920928955e-7**, largestrelativeRMS **9.0936e-8**.
  Original timeoutput wordsare counted, notonlySHA, butthefourinternalGEMM/add stages
  havenotyetbeenindividuallypaired. The exact failinginternaloperation isUnknown.
- Full50-layer nativevideo/audio drift is~0.026%/0.04% relRMS. The smalltimeδ is a
  measuredfirstchangesource, **not yet a proven sole explanation of every finaldelta**.

## Preserved failures and recovery

[R129a controls](control_summary129a.json) also completed8×3 originalforwards/3,720checks
withstableinputs/weights. Its firstcoarsetrace failed whilethefingerprintcode attempted
to encode`torch.device`; thisdiagnosticerroroccurredbefore a modelregion resultcould
becompared. [Originalsource](executed_sources129a/fingerprints129.py) and
[failedreceipt](executed_diagnostics129a/cohort_failure128.json) arepreserved.

The parent-onlyabortleftdetachedworkers. Exactprivateenv/UID/starttickswerereverified
beforeSIGTERMandthenSIGKILL aftergrace; CANNteardown tookmorethan30seconds. A later
liveprobeconfirmed nonewereleft beforekeyremoval. [Cleanup](cleanup129a.json)
records thislimitation ratherthan callingtheinitialparentabortcomplete.
No SDKinstall,driverreset,device reset or shared-process signal wasused.

`torch.device/dtype`metadataisnowcovered by [actualsmallunit](fingerprint_unit129b.json)
and [13 portable tests](test_contracts129.py). Freshroot12902/session129b neveroverwrites
a sources/receipts; theupdatedaborttracksdetachedexactprivateprocesses.
[Successful b cleanup](cleanup129b.json) confirms allprivateprocessesexited andkeyremoved,
with originalandexecutedsourcepinsunchanged. No model/activationarchivewenttotheclient.

## Next strict step

Keep#4inactive. Addnarroworiginalguarddiagnostics to pair each ofthefourHalfGEMM
inputsandFP32outputs **before** anyinplaceaccumulation, theninspect thethreeFP32
scaledadditions separately. Preserve theoldformula/returns/TP andrecord whether
extraD2H/readyfenceshide theinstability. TeststandaloneM3timeprojectionfirst; only
a causallysupported, same-scope resultcanlead to a candidatefix/full50regression.

Anychange thatalters originalarithmeticvalues mustbe reportedforuserdecision;
these nativeδvalues are **not** acceptederror thresholds. Fullperformancegoal and
this reproducibility Issue are notclosed merelybecauseafirstregionwasfound.
