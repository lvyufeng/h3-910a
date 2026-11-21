# h3-910a agent execution contract

The user requires optimization work to run through **GitHub Project + Issues**,
not only chat memory, local todos, or unlinked benchmark scripts.

## Start and finish every work session through GitHub

1. Read [the GitHub repository](https://github.com/lvyufeng/h3-910a),
   [roadmap Issue #1](https://github.com/lvyufeng/h3-910a/issues/1) and
   [Project #10](https://github.com/users/lvyufeng/projects/10), recorded in
   [the execution receipts](artifacts/diagnostics/github-execution/issue_receipts.json)
   and [the project receipt](artifacts/diagnostics/github-execution/project_receipt.json).
   Then read the current Issue, native blocked-by dependencies, recent comments,
   owner and current Project fields. Local specs are creation records, not live status.
2. Work only on a ready task. Assign the real GitHub executor; comment the hypothesis,
   inputs/config/source/weights SHA, devices/TP groups, budgets and precision gate;
   update the Project to In Progress. A ready board item is not resource permission.
3. Report measured progress, every failed attempt and restore action in that Issue.
   On a lost write response, read back GitHub state before retrying; never duplicate
   issues/projects or replay uncertain model launches.
4. End with baseline/candidate/restored timings, cold/warm/rank-max scope, actual
   kernel/copy/collective hits, memory and numeric results; link evidence and commit/PR.
   Use Keep, Reject or Awaiting approval. Reject is a completed experiment, not adoption.
   Blocked must state the concrete condition and recovery step.
5. Update the parent Issue and Project before ending. Close only when that Issue's
   exit criteria are satisfied. Do not close the overall performance roadmap just
   because a 49-step run or the project setup completed.
6. If GitHub/auth is unavailable, report it explicitly; preserve a local receipt
   but do not invent remote status or silently continue untracked experiments.

## Non-negotiable model and resource constraints

- Shared accepted reference: R124 original MiniMax-H3 plus user-accepted 910A FA.
  R126 commit `4384e89` is verified execution evidence, **not** a blanket acceptance.
- **TP stays ON**: baseline DiT TP2+USP4, live FP16 encoder TP8, all eight devices.
  The 910A physical topology is two HCCS islands 0–3/4–7; cross-island links are
  PHB/SYS CPU paths, not an eight-card full mesh. Prove real shards and physical groups.
- Preserve original 50 DiT blocks, 49 actual evaluations/50 sigma boundaries,
  FP16 matrices, original FP32 front/time/final/VAE parameters, FP32 residual/norm/
  wide-GEMM outputs, all 155 strict Half checks per step and power-of-two range guards.
  No BF16 roundtrip, HF32/TF32, clipping, student/LoRA, teacher hidden/features or skip-step.
- Any attributable numeric change: restore/disable, report numbers, request user decision
  in the Issue. Native self-repeat noise and encoded PSNR are not automatic tolerances.
  Quantization, approximate activation caching, sparse attention, fewer steps,
  PipeFusion previous-step KV and distilled checkpoints default to not authorized.
- Reuse verified on-host weights; new downloads go to the user's designated 4090 machine.
  Never download/relay model weights or activation archives to the client workspace.
- Use only explicitly claimed owned resources; CI/dormant need separate user opt-in.
  No shared SDK changes, driver/device reset, other users' processes/data deletion.
  End by cleaning experiment-private IPC/processes and releasing leases.
- Compare exact scopes: actual R126 whole-step 19.372s versus isolated replay 17.556s;
  cold combined VAE142.143s versus warm video20.565s/audio0.210s. Never substitute one.
  Encoder<2s, video VAE<20s, DiT<15s/step and actual A2/A3 comparison remain open.

Detailed rationale and sequencing live in
[the creation analysis](artifacts/diagnostics/github-execution/PLAN.md), and
[the immutable R126 evidence](artifacts/diagnostics/round126/README.md).
