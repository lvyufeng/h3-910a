# Round81 — full-model KV padding experiment

This round keeps the [verified CPU FP16 oracle](<../round78/final_CPU_reference_verification78.json>)
and the original dense weights fixed. It tests whether zero-only extra KV
output rows can remove the [small-N instability](<../round80/kv_stability_summary80.json>)
observed in the failed completeTP8 baseline. No BF16 agreement requirement is
reintroduced and no production encoder is changed.

## A hash mismatch already explained

[Signed-zero control](<embedding_sign.json>) reconstructed the real checkpoint
embedding rows and verified their hash against the fresh CPUFP16 reference.
Adding the sevenzero rank contributions in FP32 changes **exactly one `-0` to
`+0`**,with no numerical value difference. The resulting full tensor hash is
**identical to the actualR80 NPU embedding hash**. This explains the previous
embedding hash mismatch without inventing an embedding arithmetic error.
It is a local model of the reduction,not a newHCCL run or a changed teacher.
The full new driver also records actual numerical embedding differences.

## Controlled model change

[Full-model driver](<encode_fp16_padded_kv81.py>) starts from the immutableR80
baseline but uses a new private root. Its adapter changes only the two K/V
Linear call sites in the isolated candidate's text attention:

- parameters and inputs remainFP16;
- original128 output rows are copied unchanged;extra rows to256 are zero;
- extra output columns are discarded;
- Q,MLP,vision,RoPE,CPUreference and global functional operators are unchanged;
- cache setup is excluded from the warmed windows,and cache identity/prefix/
  zero-tail proofs are recorded;
- all50 layers,real vision/DeepStack and finalD2H are still included;
- full and text/visual/high-range token-subset metrics are reported,not used as
  new acceptance thresholds.

[Driver tests](<test_driver81.py>) guard fixed source/teacher hashes,new roots,
cleanup and real encode_ids timing. [Signed-zero tests](<test_embedding_sign81.py>)
prevent raw-bit mismatch being silently confused with numerical loss.
The [post-run summarizer](<summarize81.py>) requires all8 complete receipts and
matched same-call rank maxima before any timing summary.

## Resource scope and status

[Setup](<setup81.json>) records the live-owned fleet claims. The arbiter returned
two CUDA hosts before910A;both were unused and released before compute. Only
the live-leasedownedts217 is used for H3. No CI/dormant pool was requested.

The KV adapter was completed by a fresh fork and reviewed before execution.
[Root regression checks](<root_review81.json>) corrected K/V shared-storage
cache identity and bound-method cleanup without changing the candidate or
teacher. The first attempt returned a diagnostic-only Python manifest-index
shadowing error after its first full forward. Its [log](<run.log>) and
[failed state](<failed_state81.json>) are preserved. The
[corrected new-root driver](<encode_fp16_padded_kv_fixed81.py>) renames only that
manifest variable and completed in a distinctwrite-once root;no uncertain
command was replayed. [Retry diagnosis](<retry_diagnosis81.json>) pins the change.
No full-model KV-padded precision or speed pass was claimed before collecting
all8 actual receipts. The corrected run now completed exit0;all8 full receipts
and source/input hashes were independentlychecked in the
[actual summary](<summary81.json>) and [runtime tests](<test_actual_evidence81.py>).

## Actual complete-encoder result

- **Repeat stability fixed in this graph:** all8 first/second outputs and each
  rank's3 warmed `encode_ids` outputs are bit-identical.
- **Full encoder latency:**2.234800s /2.155799s /2.213749s same-call rank maxima,
  median **2.213749s**,including realvision/DeepStack/all50textlayers/finalD2H.
  This remainsabove2s;it isnotavision-onlytiming or externallyconditionedtexttime.
- **Precision still pending:** completeCPUFP16 selected50 relativeRMS7.4201%;
  text27nonvisualtokens0.0739%,visual2016tokens7.8914%,reference-low-range2033
  tokens10.5391%.The gap isnot solely high-rangeoutliers,and no tolerance is
  chosen afterthefact to call it apass.
- **Embedding:**actualnumericRMSzero onall8ranks,consistentwith thesigned-zero
  diagnosis above.
- **KV cache:**100Halfzero-paddedweights,262,144,000B perrank,alloriginalprefixes
  exact/alltailszero/600actualpaddedcalls. Source-onlyauthoredmetadata inadapter
  snapshots retains`not_executed_here`;theactualrankreceipts aretheexecutionproof.
- [Post-run source/worker check](<post_probe_verification81.json>) confirms23
  production and13API sourcesunchanged,noNPUworkersremaining. Theolderverifier's
  hardcodedCPU-PIDfieldwasnotusedasfinalproof;[exact-command cleanup](<final_cleanup81.json>)
  checksallCPU/NPUworkercommandsandtheoriginalCPUcompletionreceipt. Theowned910A
  leasewasreleased;thetwo unusedCUDAgrantswerealreadyreleased.

[State](<STATE.json>) records allcollected jobs andnoncompletion.
[Next experiment](<NEXT.md>) isolatesCPUgoldvisionconditioningfromrealNPUtext
arithmetic beforechoosingnewbias/TPFP32-epiloguekernelwork. Itmustnotbeusedas
external-conditioningwholeencoderperformanceevidence.

The [future GEMM pattern](<future_gemm_pattern81.json>) only identifies a
FP16-operand/FP32-accumulator declaration in a design header;it is not claimed
as an already compiled or benchmarked implementation. Native biasedLinear
rounding and TP partial-output rounding still need actual kernel work if their
full-model CPUFP16 discrepancy remains.
