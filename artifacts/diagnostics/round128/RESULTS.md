# R128 outcome — native repeat gate blocked before copy-stream candidate

Live [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) is **Blocked** by
[new P0 Issue #15](https://github.com/lvyufeng/h3-910a/issues/15).
No runtime/copy-stream optimization or numeric tolerance was promoted.

## Actual execution

[Code087c5fd](https://github.com/lvyufeng/h3-910a/commit/087c5fd),
[exact executed sources](executed_sources128a/integration_hooks127.py),
[actual summary](formal_summary128a.json) and
[original request receipt](executed_formal128a/engine_overlap128a/lifecycle_result127.json)
are preserved. Later semantic fixes are not passed off as executed code.

- The formal engine initialized in239.040s. Its same8 workers completed one
  **original50 blocks×49 evaluations**, **60,760 strict Half checks**,124frames at1344×768/24fps,
  and original native Float audio. Actual TP2 rank pairs were0–1/2–3/4–5/6–7;
  localheads28 before USP4. EncoderTP8/FP32 VAE/all original protections stay native.
- At samplingstep25, extra **three** native50-block replays used a clone of this
  request's own live NPU forward input, not teacher/GPU features. Each replay had155checks;
  **3,720 additional checks** across8 ranks are explicitly not counted as sampler steps.
- The saved original prediction bytes were unchanged after replay on all8 ranks;
  replay outputs were never fed into sampling. Original request generation completed.
- Generation wall1524.562s includes extra replay, fixture saving and complete CPU-master
  hashes. It is **not** a candidate speed comparison with R1271339s. The original
  whole-forward warm rank-max median was19.836s, still not<15s/step.
- **Neither serial nor async model adapter was ever installed.** Their full-size
  master/104-wide-leaf proof and actual model stream overlap did not execute.

## Unchanged native same-input deltas

See [rank0 original gate](formal_replay_rank0128.json) and
[rank-max repeat findings](native_repeat_failure128.json).
Against the first native replay:

- Second native replay,19.855576s: video relRMS **0.0261953%**, maxabs **0.0168597**;
  audio relRMS **0.0411016%**, maxabs **0.00865376**.
- Third native replay,19.855652s: video relRMS **0.0263200%**, maxabs **0.0253544**;
  audio relRMS **0.0357664%**, maxabs **0.00477457**.
- Both raw output leaves were not bit-identical on all8 ranks. All original CPU
  master hashes remained unchanged. These are unmodified native-replay findings,
  **not attributable copy-stream errors** and **not allowable candidate tolerances**.
- Input mutation across calls, non-parameter forward/workspace state and the first
  changing operation/collective remainUnknown. Do not name FA/encoder/HCCL/allocator
  as root cause merely from the final deltas. Issue#15 now requires that isolation.

## What the small gates proved — and did not

[Owner-thread test](worker_stream_unit128.json): actual distinct owner-thread copy,
wrong-thread refusal before NPU access, all65536 Half patterns/transpose/offset raw
bytes preserved. Synthetic async copy/body intersection2.867ms, serial controls0;
this is not model or HBM/kernel-busy acceleration evidence.

[Refreshed native50-hook rehearsal](native_hook_unit128b.json): small synthetic50
blocks, all output/master words equal, bounded detached-storage retirement, missing
first-block recovery and injected block20 rollback passed. No R126 resident/reuse pool.

[Eight-rank status smoke](cohort_unit_rank0128.json): rank3 injected setup error was
refused by all8 before another collective; HCCL then worked. It does not promise
recovery from fatal NPU/HCCL faults. Setup/status uses the existing TCP Store; a
private inotify failure monitor prevents rank-local failure from stranding peers.

[Portable tests](test_contracts128.py) now24, including the semantic fix below.
Only23 were run before this original experiment; this distinction is retained.

## Driver semantics and cleanup

The executed driver used `restored` to mean **output bit equality**, then treated
that false value as a hook-restoration failure at its final gate. Since no adapter
had been activated, this produced an **AssertionError exit1 after the original49
request succeeded**, not a failed original49 computation or an allocator error.

The later code separates `native_hook_state_verified` from
`restored_output_bits_equal` and classifies native-repeat failure asBlocked rather
than requesting permission for an untested candidate. This fix was unit-tested
but **not rerun on the model to hide/replace the failed receipt**.

[Cleanup](cleanup128a.json) confirms every exact private process exited, private
IPC key removed and original/private launched source/noise hashes matched. The
owned ts217lease was released. Independent queued NUMA/H2D probing was canceled,
not executed, until the baseline gate is understood. No model/activation archives
were downloaded to the client; only bounded compressed JSON/text evidence.

## Next action

Keep the private stream implementation **inactive**. Issue#15 first checks input
and state invariants, then original wide GEMMs/FA/FP32 norm-residual/TP-USP collectives
in separate paired diagnostics to find the first changing result. If a real numeric
policy change is necessary, report its numbers for the user's decision. Do not
relax the bit gate simply because this native drift is relatively small.

Issue#4 remains unfinished; overall latency goals and actual A2/A3 comparison stay open.
