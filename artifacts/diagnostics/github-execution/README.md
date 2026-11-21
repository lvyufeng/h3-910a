# GitHub-first execution: MiniMax-H3／910A

Live tasks: [Project #10](https://github.com/users/lvyufeng/projects/10) ·
[roadmap #1](https://github.com/lvyufeng/h3-910a/issues/1).
The Project is private to its owner; repository Issues and published evidence are public.
GitHub is the source of live status. Local specs/receipts are **creation and verification records**,
not a second task board. Do not rerun initialization to reset later execution state.

## Execution order

- Completed reference: [baseline #2](https://github.com/lvyufeng/h3-910a/issues/2).
- Completed observation: [formal engine cold/warm/reload #3](https://github.com/lvyufeng/h3-910a/issues/3).
  [R127 findings](../round127/RESULTS.md) verify three same-worker requests and182,280checks;
  **not** a performance/precision promotion. Next is
  [safe asynchronous H2D/lifecycle #4](https://github.com/lvyufeng/h3-910a/issues/4).
- Main kernels: [FA #5](https://github.com/lvyufeng/h3-910a/issues/5),
  [GEMM #6](https://github.com/lvyufeng/h3-910a/issues/6).
- Decoder: [tile microbatch and chunk schedule #7](https://github.com/lvyufeng/h3-910a/issues/7).
- Capability work: [regional graph #8](https://github.com/lvyufeng/h3-910a/issues/8),
  [TP/Ulysses/Ring topology #9](https://github.com/lvyufeng/h3-910a/issues/9),
  [live TP8 encoder #10](https://github.com/lvyufeng/h3-910a/issues/10).
- Default not executable: [numerical/approximation decision #11](https://github.com/lvyufeng/h3-910a/issues/11).
- Final: [49-step integration #12](https://github.com/lvyufeng/h3-910a/issues/12),
  [honest A2/A3 hardware comparison #13](https://github.com/lvyufeng/h3-910a/issues/13).
- Follow-up reliability risk: [native CPU Half packing SIGSEGV #14](https://github.com/lvyufeng/h3-910a/issues/14).
  R127c succeeded, but the earlier native initialization−11 root cause remainsUnknown.

Each child has native GitHub sub-issue and blocked-by relationships, plus objective,
measurement scope, work list, acceptance gate, source links and rollback requirements.
The main roadmap stays open until its actual goals or the user's changed goal are satisfied.

## Board fields and views

Status: **Backlog, Todo, In Progress, Blocked, Awaiting approval, Done**.
Fields: Priority, Component, Phase, Precision gate, Depends on, Execution order.
Views: **执行看板**, **执行顺序**, **等待精度决定**.

`Done` means the experiment has a supported conclusion (Keep or Reject), not that
all numerical changes were accepted. A rejected or unsupported candidate must not
be selected into integration. An optional numerical branch is not a dependency of
the lossless main line. Follow native dependencies before moving a task to In Progress.

## Agent workflow

[AGENTS.md](../../../AGENTS.md) and the always-on
[Cursor rule](../../../.cursor/rules/github-execution.mdc) enforce:

1. Fetch roadmap/current Issue, comments, native dependencies and Project fields at session start.
2. Claim a ready Issue, assign the actual GitHub executor, comment hypothesis/input/source/device/
   precision/compute budget, and mark In Progress. Live-claim owned hardware separately.
3. Run baseline→candidate→restored, preserve actual operator evidence and all failure receipts.
4. Comment timings with exact scope, actual hits, memory and numeric results; link commit/PR.
5. Use Keep/Reject/Awaiting approval; restore on numeric changes and request user decision.
6. End by updating Issue, Project and parent; record next step, cleanup and release lease.

Do not create a scheduled automation or unapproved compute resource merely because a task is ready.
A2/A3 access, model changes or reduced precision still require their own applicable authorization.

## Command examples

Read live work and comments:

```bash
gh issue view 1 --repo lvyufeng/h3-910a --comments
gh issue view 3 --repo lvyufeng/h3-910a --comments
gh api repos/lvyufeng/h3-910a/issues/3/dependencies/blocked_by
gh project item-list 10 --owner lvyufeng --limit 100 --format json
```

Only **after** checking readiness, assign and start the actual task:

```bash
gh issue edit 3 --repo lvyufeng/h3-910a --add-assignee @me
gh issue comment 3 --repo lvyufeng/h3-910a --body-file /path/to/experiment-contract.md
```

Project item/status field IDs and option IDs are saved in
[the project receipt](project_receipt.json). Read live field definitions before using them:

```bash
gh project field-list 10 --owner lvyufeng --format json
gh project item-edit --project-id <project-id> --id <item-id> \
  --field-id <Status-field-id> --single-select-option-id <In-Progress-option-id>
```

For a lost GitHub write response, read back by issue marker, number or item ID before any retry.
The setup encountered one lost dependency response; it was reconciled through native GETs,
not blindly replayed, and no duplicate Issue was created.

## Reproducible setup and tests

- [Original analysis](PLAN.md): model/precision/resource constraints, R126 evidence and rationale.
- [Spec](roadmap_spec.json): initial issue definitions and 21-edge dependency DAG.
- [Issue receipts](issue_receipts.json): exact GitHub IDs, URLs and creation verification.
- [Project receipt](project_receipt.json): exact Project/field/item/view IDs, linking verification.
- [Issue initializer](create_issues.py) and [Project initializer](create_project.py):
  no writes without `--apply`; reconcile before creating, do not touch unrelated Projects.
- [Contract tests](test_contracts.py): stdlib only, no GitHub writes or model execution.
- [Live verifier](verify_execution.py): read-only check of actual Issue IDs, native dependencies,
  Project membership, seven fields, repository linking, views and ready tasks; not a performance gate.
- [Verified setup receipt](verified_github_setup.json): 13 actual Issues, 12 native sub-issues,
  21 dependency edges and 13 Project items verified; this is a dated snapshot, not live status.

```bash
python3 artifacts/diagnostics/github-execution/test_contracts.py
python3 artifacts/diagnostics/github-execution/verify_execution.py --require-project
```

GitHub CLI requires repository and `project` scopes. Keep credentials in the system
keychain; never commit tokens or downgrade to plaintext without a separate explicit decision.
If scopes cannot be saved by a background shell, refresh from the user's own terminal:

```bash
gh auth refresh -h github.com -s project
```

This setup does not adopt a new precision policy, reduce the 49-step count, disable TP,
start a scheduled task, claim hardware or declare the performance objectives achieved.
