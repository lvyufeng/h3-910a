# R129 — original native repeatability diagnostics (Issue #15)

Live [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) is In Progress;
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) remainsBlocked and copy candidates off.
[Initial contract](issue15_start.md) preserves TP2+USP4 / encoderTP8 and155strict checks.
This is active debugging, not a latency or new precision-policy acceptance.

## Preflight facts

[On-host native-input inspection](input_preflight129.json) verifies all8 saved
R128ownlive input SHA matches,~44.68MB each; no client model/activation transfer.
Text/video/audio positions have no duplicates or cross-modality intersections.
Thus duplicate-position Half `index_add_` is not supported by these inputs; other
input/state/operation changes remainUnknown.

[Small tensor-hash unit](fingerprint_unit129.json) verifies scalar,empty,transpose
and inference-mode tensors; [portable tests](test_contracts129.py) cover the private
RPC/no-generate contract, exact session/root, original155checks and trace returns.
A first CPU-onlyprobe missed sourcing existing CANN env and failed `libhccl.so`
import; corrected shell succeeded without SDK/install change. A gitlsremote SSL
connection timed out, while GitHubAPI verified published51db4b4 andlive#15claim.

## Private implementation

- [Native diagnostic RPC](diagnostic_rpc129.py): actual DiffusionWorker instance's
  unique stringmethod, original graph on same native thread/device. Reuses prior own
  NPU input; checks pre/post inputvalues and native host/resident parameter/bufferstate.
- [Control driver](generate_diagnostics129.py): oneformal Omni, control RPC only,
  **no generate or sampler steps**. All8 execute; rootcontrol reply is aggregated,
  rankreceipts separately prove execution.
- [Original bootstrap](integration_hooks127.py) preservesacceptedR124graph+FA,
  FP32exceptions and155checks; onlyprivateclass/method registration, notsharedSDK.
- [Region trace](region_trace129.py): original pre/post boundaries and output SHA,
  originalreturns preserved; traceD2Hsync is explicitlynonclean andmay hide a race.
  A hashdifference locatesa boundary but isnot completewordcomparison or kernelrootcause.
- [Actual FA activity](fa_activity129.py): originalwrapper/ACL call counts,
  metadataanderrorbranch observations, nofallbackpolicychange or cachesclear.
- [Fingerprint helpers](fingerprints129.py) distinguishlogicalwords from ptr/versionmetadata;
  no full50activation retained. [Pinned launch](launch_source_pins129a.json) and
  [private no-generation launcher](launch_diagnostics129.py) guard against uncertain replay.

## Current execution

Private session `engine_repeat129a`, PID3771945, initializedfromon-host originalcheckpoint.
Three diagnostics: sameallocation/freshclone nativecontrol, coarsefirst-change SHAtrace,
then detailed blocktrace. No nativecopy candidate installed, no model49generationforlogs.
Source snapshots are onlytext; all weights/inputs/activationstayon-host.

Wait for completed receipts to claim anyfirst-changing region. Individual numericnoise
is not a candidate tolerance. Any laterfix requirescausalpairing and explicitquality
policy if arithmetic changes; #4 cannotadvancejustbecause thisdiagnostic codeexists.
