# Round75 — actual vision output-column and raw-gather entry probes

## Scope: useful exact primitives,not another complete encoder speedup

This round tests four **current experimental L0** vision Linears and fixed-size
raw-word gather primitives while Round74's independent original-BF16 HF reference
runs. It does not integrate a head-parallel tower,change the current encoder,
close the independent quality gap,or claim encoder≤2s.

[Capture summary](<capture_summary.json>) verifies8 excluded full encoder outputs
and all intermediate identities against Round73,original inputs/parameters
unchanged. Rank0 captures full BF16 x/weight/bias/output for qkv/proj/fc1/fc2:
**317,946,420 fixture bytes**,under128MiB per file and384MiB total. All actual
operands are finite. Later layers and all eight merger Linears are **untested**.

The [capture driver](<capture_linear75.py>) restores attention/RoPE/Linear wrappers
and clears shadows. Capture/compile/I/O time is excluded from performance claims.
The endpoint remains stable experimental SHA`67ef196e…`,not original HF identity.

## Output-column result and retained counterexample

[Column summary](<column_summary.json>) tests every original K input value,
original fused bias and BF16 output interface. It does not split K or sum row
partials. Both **private FP32 values** and **public BF16 outputs** are measured,
then all8 pieces are explicitly inverse-indexed into the complete output matrix.
Each candidate has three independently computed repeat proofs.

Passing current-L0 shapes:

- QKV:K=1152,local output432(Q/K/V each two72-wide heads),all8 partitions.
- Proj:K=1152,local output144,all8 partitions.
- Fc1:K=1152,local output538,all8 partitions.
- Fc1 zero-output-row pad538→544,crop538:all8 partitions.

All four combinations match complete FP32/BF16 matrices and three repeats.
[Plan](<column_plan75.py>) maps QKV as three outer groups,each with rank's two
heads. Simple rank-major concatenation is wrong:the deliberately wrong QKV
reconstruction is rejected. No output rows are omitted or duplicated.

**Fc2 at K=4304/output144 fails** and remains rejected:

- Partition0 versus full FP32:908,200 differing words,maxabs3.8147e-6.
- Its BF16 interface: **354 differing words** out of1,161,216,maxabs0.0078125.
- The same narrow shape repeated produces **26/17/32 differing BF16 words**
  relative to its first call;full-width repeat controls remain exact.

[Original numeric failure](<column_determinism_failure02.json>) is retained.
Failure is shape-specific:proj also uses144 output rows but K=1152 and passes.
This does not prove all N=144 shapes are invalid,nor identify the backend's
internal split-K/atomic heuristic as a measured mechanism.

A separate [padding experiment](<fc2_padding75.py>) and[receipt](<fc2_padding01.json>)
keep all K=4304 values and fused bias,adding explicit **zero weight/bias output
rows**,then crop to144. Widths **160/192/256/512** each pass all8 private FP32/
BF16 full reconstruction and three repeats.160 is the smallest **tested** passing
width,not a universal lower bound or proof of deterministic execution for all
future inputs. The failed144 family is not reclassified as accepted.

[Aggregator](<summarize_column75.py>) binds padding recovery to the **same actual
fc2 input/weight/bias/output fixture SHA** as the retained failure,and verifies
mode names against real widths. Recovery from an unrelated fixture or a bogus
padding label is rejected. No numerical repair or precision flag is introduced.

## Original serialization failure also retained

[Initial probe](<column_probe75.py>) failed before math:the actual saved payload
included the installed numeric`torch_npu.npu._format.Format` enum in layout
metadata,so`weights_only=True` refused it. [Failure receipt](<column_pickle_failure01.json>)
is preserved and not conflated with the later fc2 numerical failure.

A new [restricted probe](<column_probe_safe75.py>) uses a new run root,checks the
trusted own capture file SHA first,and allowlists **only the installed Format
class** in a scoped`torch.serialization.safe_globals` context. It does not use
`weights_only=False` or unrestricted persistent global allowlisting. Every loaded
operand still checks shape,dtype,raw tensor SHA and original checkpoint bytes,
and full Linear replay must reproduce the captured BF16 result.

Next capture code should serialize numeric layout formats as plain`int` to avoid
this dependency. Existing executed captures and failures were not overwritten.

## Genuine eight-card raw gather:direct BF16 already works here

Two separate [gather runs](<wire_probe75.py>) use the **same explicit experimental
encoder GroupCoordinator construction**,not the different production vLLM device
communicator. Six fixed equal-size payload shapes and rank-major/row-major
concatenation are tested on8 real NPUs with HCCL:

`(128,544),(2016,432),(8064,432),(8064,144),(8064,538),(8064,544)`.

[Transport summary](<wire_summary.json>) validates:

- **Direct BF16 gather**:all65,536 possible BF16 encodings preserved exactly,
  including signed zero,subnormals,finite extrema,Inf and distinct NaN payloads.
- **INT32 pair-view gather**:the same result,with identical cross-mode source/
  expected gathered payload identities. INT32 pair transport is therefore
  **not required for this direct BF16 gather path**.
- Per mode **384 complete gathered tensor proofs**,48 raw device-view roundtrips
  and24 bad-layout guard rejections. All counts derive from validated receipts.
- All16 rank receipts match source/rank/device/backend,raw source readonly,ND
  format0/2,zero offsets and32B pointer alignment. Every rank's distinct tag and
  payload appears once,in correct order.

[Raw view helper](<raw_wire75.py>) performs metadata`view(dtype)`,not numeric
conversion. Proofs copy **INT32/INT16 raw words** to CPU,never BF16→FP32 or a
BF16 CPU conversion that could normalize NaN payloads. Odd BF16 widths,odd BF16
offsets and noncontiguous inputs are rejected. INT32 width269 is valid:it decodes
to even BF16 width538. Guard negatives are tested at128 rows on each rank;this
is not universal guard coverage over every shape.

A source premise was corrected:in this GroupCoordinator **`all_reduce`** uses
BF16→FP16 promotion,while **`all_gather` does not**. The current encoder itself
uses FP32`all_reduce` for partial projections/embedding,not this newly measured
all-gather primitive. These gather proofs do **not** certify the encoder's
all-reduce arithmetic,production vLLM communicator,BF16 compute ISA or whole
end-to-end quality.

## Timing and implementation limits

Single-card sequential eight-column probes are **not TP8 speed measurements**.
Part medians cannot be divided into full-projection time to claim complete-stage
speedup;doing eight pieces sequentially may be slower than one full projection.
A future real TP8 candidate must time same-call rank maxima,include all required
communication,and pass full encoder/intermediate identities on actual inputs.

Raw-gather warm samples have a substantial dispatch/synchronization floor and
were collected while the two CPU reference jobs shared the host. Their size/time
relationship is not cleanly monotonic and the dtype modes run in separate roots.
No INT32-versus-BF16 speed/overhead or bandwidth superiority is claimed. The
bare medians are retained diagnostic observations with cohosting scope.

Proj/fc2 retain full K in these **single-operation probes**. A truly partitioned
upstream head/fc1 implementation must first gather full inputs for them or use a
different reduction scheme. These receipts do not prove a zero-communication
partitioned vision tower. The existing Round73 strided-cast kernel assumes16
heads and pitches1152/3456;head-parallel shapes would require a separately
verified new kernel,not an edit of an already accepted source.

## Final verification and next gate

[Final fingerprints](<final_verification.json>) verify23 unchanged production
sources,35 executed probe/dependency sources(including rejected and recovery
runs),and five real fixture/reference files. All match. Missing remote tensor
blobs are not claimed locally rehashed;local source/manifest comparisons are
separate from the remote streaming hash proof.

[Cleanup](<cleanup_receipt.json>) records idle NPUs,zero accelerator processes,
completed probe jobs and released owned-host lease(shared with Round74's completed
CPU references). No production source was edited or deployed.

16 new source/actual-transport/geometry gate tests pass alongside195 earlier
regressions,**211 total** in[regression receipt](<test_receipt.json>). They reject
CPU/non-HCCL/mixed-mode/bad-word transports,false production scope,different
padding fixtures and bogus padding labels;retain the original pickle/narrow-GEMM
failures;pin raw view constraints,all65536-word gather evidence,current complete
encoder identity and unchanged source/reference fingerprints.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round75 -p 'test_*.py' -v
python3 artifacts/diagnostics/round75/summarize_capture75.py
python3 artifacts/diagnostics/round75/summarize_column75.py
python3 artifacts/diagnostics/round75/summarize_wire75.py
```

This establishes exact candidate primitives and a retained narrow-GEMM rejection.
It does not prove all116 vision Linears/later inputs,an integrated tower speedup,
independent HF equality,full encoder≤2s or the overall lossless objective.
