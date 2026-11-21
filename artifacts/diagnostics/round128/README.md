# R128 — Issue #4 private worker-owned native copy stream

Live task: [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) ·
[Project #10](https://github.com/users/lvyufeng/projects/10).
See [initial contract](issue4_start.md). The task is now **Blocked** by
[Issue #15](https://github.com/lvyufeng/h3-910a/issues/15): unchanged native same-input
replay was not bit-repeatable **before any copy-stream adapter was enabled**.
See [actual outcome and next gate](RESULTS.md); there is no acceleration/precision promotion.

## Implemented, not globally enabled

- [Owner capability](owner_stream128.py): actual execution-thread creation,
  PID/nativeTID/device/current-stream checks, strict real H2D warmup, retained probe,
  distinct private handle and explicit owner-only drain/close; no global factory switch.
- [Native50 adapter](native_copy_adapter128.py): changes only existing DiT hooks,
  calls original native prefetch/raw-copy/binding, keeps real detached storage references
  until consumer-complete, bounds retired blocks to2 with counted backpressure.
  There is **no R126 two-slot pool, new residency or new model math**.
- [Formal same-input gate](formal_replay128.py): replay inputs originate from the live
  NPU request at step25, not teacher/GPU features. Extra evaluations are separately
  counted and never passed to the sampler. Native→serial→async→restored root outputs
  must be raw-word equal; paired104 wide-GEMM leaves use the original algorithm and
  exact native master mirrors on the same input. Full physical masters get byteproof.
- [Private original graph integration](integration_hooks127.py) and
  [formal driver](generate_lifecycle127.py) keep original50×49/155checks and TP2+USP4,
  live encoderTP8 and FP32 VAE. Only after exact/support gates and≥1% same-scope gain
  will full request2 use privatecopy; request3 restores nativecopy. A declined gate
  terminates cleanly after the original first request, not a fake candidate run.
- Some filenames retain127 because the executed original bootstrap/observer interface
  is reused. All new runtime paths are private `/tmp/h3_overlap12801`, with isolated IPC.

## Actual completed small gates

[Thread unit](worker_stream_unit128.json): distinct owner-thread handle, wrong-thread
use refused before runtime access, all65536 Half word patterns plus transpose/offset
copies exact; synthetic serial overlap0, async overlap2.867ms, restoredserial0.
The first serial had cold kernel startup, so its229ms is **not** compared to6.5ms
async as a supposed35× model speedup. This is not formal/H3 performance evidence.

[Native hook rehearsal](native_hook_unit128b.json):50 small synthetic blocks,
all50 outputs and native physical master words equal in serial/async, native restored
forward equal; missing first-block recovery and injected block20 exception rollback
passed. Both modes' inclusive copy/body overlap was0 for these tiny operations.
The adapter's `copy_stream` in serial rehearsal facts names the **warmed but unused**
private capability; [formal profiler](formal_replay128.py) additionally reports the
actual selected hook stream. Do not confuse capability existence with its use.

[24 portable tests](test_contracts128.py) cover ownership, no global switch, recorded
fences, bounded backpressure, non-trajectory accounting, live-input provenance,
exact session import and conditional candidate generation. Hardware/model/precision
gates are separate. [Eight-rank TCP status smoke](cohort_unit_rank0128.json)
verified a rank3 injected setup failure was refused by all8 ranks and HCCL remained
usable afterwards; it does not claim recovery from a fatal runtime/collective fault.

Prelaunch review caught a legacy session-prefix assertion and rank-local failure
paths. The [preliminary pins](launch_source_pins128a.json) were invalidated **before
any model launch**, not treated as execution proof. [Final pins](launch_source_pins128a_final.json)
include exact session guards, transactional104-leaf activation and
[cohort setup/return coordination](cohort128.py), plus an event-driven launcher
monitor that aborts only its verified private process group on a fatal rank error.
The [first small-test sources](executed_synthetic128a/native_copy_adapter128.py) are
preserved with SHA matching their earlier executed receipt.

## Still required

[The native-only formal attempt](formal_summary128a.json) completed original8×49
and60,760checks plus3,720 separately counted replay checks, but no serial/async
model adapter was activated. The same-input native video/audio prediction drift
is0.0262%/0.0411% relRMS; it is **not a permitted candidate tolerance**.
The exact launched sources and failed driver receipt stay archived; only later
code fixes distinguish physical restoration from output bit equality. No repeat
inference was launched to conceal that exit1. Own process/IPC cleanup and lease
release are complete; queued NUMA probing was canceled, not executed.

Actual formal-worker same-input bitproof, real TP2 parameter layout/source pins,
clean rank-max median, memory, copy/consumer waits and full cold/reload scopes.
NUMA is separate, not changed alongside stream selection. An inclusive prefetch
span includes allocator/host gaps and is not pure DMA or kernel-busy occupancy.

Native CPU Half initialization−11 remains an unresolved [Issue #14](https://github.com/lvyufeng/h3-910a/issues/14).
Source/input metadata and failed receipts must be preserved before any edits/retry.
No local model/activation archives, no shared SDK/driver reset, no numeric promotion.
