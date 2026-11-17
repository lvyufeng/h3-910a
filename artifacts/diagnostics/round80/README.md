# Round80 — CPU FP16 oracle verified; operator differences isolated

## Independent reference is now complete

The original CPU job finished with exit0 and was collected without replay.
[Final CPU verification](<../round78/final_CPU_reference_verification78.json>)
confirms two complete **27-layer vision +50-layer text** runs agree bit-for-bit
at H3's pre-finalnorm boundary and at the HF post-finalnorm output. Inputs and
903 converted FP16 parameters are unchanged;the12 used checkpoint shards match
[the pinned official model](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/text_encoder)
LFS hashes. The oracle is original HF CPUFP16 with its declared originalFP32
islands—not oldBF16 output casting,not a FP32 Linear teacher adapter.

[Reference receipt](<../round78/reference.json>) stores pre-finalnorm output
shape `[1,2043,5120]`,FP16,max magnitude19216.0,rawSHA`82944620…`.
First/second CPU compute1811.774/1798.312s are reference cost,not NPU performance.
The CPU oracle is verified;**NPU precision and the whole optimization objective
are not automatically accepted**.

## Same-input operator controls

[Leaf receipts](<operators.json>),[factor controls](<factors.json>) and
[validated summary](<summary80.json>) separate several real numeric effects:

- **Biased FP16 Linear:**native NPU biased output is bit-identical to native
  no-bias FP16 GEMM followed by FP16 bias addition. That extra rounding explains
  most of the same-input biased Linear gap (relativeRMS3.61e-4 vs9.75e-6 without
  bias). An explicitFP32 bias/GEMM diagnostic proxy reduces it to9.18e-6,
  but needs FP32 cached weights and is not our final fused kernel.
- **LayerNorm:**ordinary random rows differ by7.77e-6 relativeRMS,not12.5%.
  The large aggregate error came from a deliberately constant high-offset row;
  native NPU produces about0.99 max error there. A shifted stable decomposition
  fixes that constant row and matches the FP64 diagnostic,while the stable
  reduction probe itself was not bit-stable. It cannot be silently adopted.
- **GELU:**the tanh version is not a large mismatch in this control. Extreme
  sentinel values inflate the RMS denominator,so its tiny relativeRMS is not a
  claim of bit equivalence. The merger's originalerf variant is not changed to
  tanh merely to lower error.
- **FP32 attentionPV:**arbitrary FP32 probabilities have2.06e-4 relativeRMS
  error vsCPU. The [process-local policy control](<policy.json>) found
  `matmul.allow_hf32=False` by default,and forcingFalse/True did not change the
  error. The initial flag-only hypothesis was rejected;the process restored
  its originalflags and no machine-wide configuration was changed.
- **FP16-exact compensatedPV:**splitting probabilities into high and scaled
  residual FP16-exact operands lowers the same-input PV gap to2.99e-7.
  [Split receipt](<pv_split.json>) records nonzero probability reconstruction
  error;this is not a blanket exactness theorem or free acceleration.
- The actual `npu_grouped_matmul` FP16-input/FP32-output probe was rejected by
  the device (`y dtype should be FLOAT16`). A schema's output-dtype argument
  does not prove910A support. Future FP16→FP32 partial-GEMM AscendC work remains.

## Whole-vision factor A/B and restored baseline

[Full vision factors](<vision_comp.json>) keep the same CPU teacher and the same
FP16 parameters:

| Factor | CPUFP16 merged relativeRMS | Resident vision median |
|---|---:|---:|
| Native before |1.4568% |1.4546s |
| FP32 biased-Linear diagnostic cache |1.2887% |1.4394s |
| Above + compensatedPV |1.1815% |2.2484s |
| Above + stable LayerNorm |1.2105% |2.2724s |
| Native restored |1.4568% |1.4546s |

Native restoration is bit-identical,and parameter/input hashes are unchanged.
The compensation improves some numerical error but loses the2s vision budget
by itself;it is **not deployed or called successful**. FP32 cached bias weights
consume2,362,750,400extra bytes. These are diagnostic controls,not optimized
production kernels or complete-encoder performance.

## Complete TP8 candidate

The [full TP8 driver](<tp8/encode_fp16_tp8_80.py>) imports the isolatedR79
candidate and pins the verified CPUFP16 stage files. Text attention uses a
scoped FP32 MATH-like helper with FP16 I/O and no±50000clamp;no global functional
replacement. It reports TP partial-output rounding as a known deviation.

The root [corrected a fresh-fork shape claim](<audit_corrections80.json>) before
execution: bare text tensor still has its leading batch,so `_encode`'s `[0]`
correctly removes that batch. We did **not** invent a production shape fix.
The driver probes private-candidate `encode_ids` and uses it for warm timing if
its result matches the explicit diagnostic path.

The first bounded TP8 attempt completed all50 layers and returned exit1 after
**repeat instability on every rank**. [All8 failure receipts](<tp8/failed_summary80.json>)
show the same first output across ranks,finite FP16 values,correct positions and
visual masks,but7.5698% selected50 relativeRMS versus CPU and0.9011% between
first/second passes. Warmed encoder timing was not reached. The
[original log](<tp8/run.log>) and [process reconciliation](<tp8/failed_run_state80.json>)
are preserved;allworkers exited. The run was not replayed in its write-once root.
A [same-input storage-offset control](<offset.json>) further found the small
N128 GEMM remains slightly non-bit-stable even afterzero-offsetcloning;theN3456
control was stable. Therefore we do not claim thatclone() alone fixes the failure.
This failure is distinct from the verified CPU reference,and no complete-encoder
precision pass or warmed2s result is claimed.

The [zero-row-padding control](<kv_stability_summary80.json>) then retained the
sameFP16 inputs/weights but changed only the GEMM output tiling: originalN128
matched0/8 repeats,while zero-padding toN256/N512/N1024 matched8/8 each. This is
a concrete next candidate for the full-model retry,**not yet a complete-encoder
fix**;CPU relativeRMS is1.46e-5 afterpadding vs1.05e-5 unpadded.

[State](<STATE.json>) records all collected operator jobs and the failed fullTP8
run. [Final cleanup](<final_cleanup80.json>) confirms23 production and13 API
sources unchanged and no remainingCPU/NPU diagnostic workers. The ownedhost
lease was released. Full encoder2s,full FP16 numeric agreement and the official
A2/A3 comparable whole-pipeline objective remain active work.
