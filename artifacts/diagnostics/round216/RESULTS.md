# R216 — synchronization prototypes failed before a valid measurement

No R216 candidate output or performance result exists. Default production files were not changed. Later [R217](../round217/RESULTS.md) supplies an actual compiled-and-executed follow-up, including runtime faults, numeric comparisons and restored baseline.

## Failure inventory

| Attempt | Actual failure | Evidence |
|---|---|---|
| High-level matmul under `disable_sync=True` | `Don't support high level api in new_stmt_scope with disable sync` | [Context error](PIPE_ALL_context_error.txt) |
| C_l0c moved outside its original scope | 155,648 B requested, only106,496 B remains | [Scope allocation error](disabled_scope_error.txt) |
| Direct official generator with C_l0c still outside | Same L0C allocation failure | [Executor allocation error](low_level_executor_error.txt) |
| Initial direct-generator import | `MatmulApiCls` is in `tik_cube_api`, not `matmul`; ImportError | Preserved transcript, corrected in [current failed prototype](tik_ops_utils_barrier.py) |
| Shared-L0C source generator | Exact replacement not found at `old_mm`; assertion prevented generation/upload | [Failed draft generator](make_l0c216.py) |
| Earlier textual edit | Newline missing before `if N1MN0_to_MN`; SyntaxError; subsequently fixed | No successful kernel before/after this fix |

The claim that a shared per-q-block L0C implementation was ready was premature. [Forward draft](flash_attention_fwd_l0c.py) is only a copy of the original; the shared-L0C generator never completed. Do not run these drafts as working optimized kernels.

## API facts established from installed sources

- `MatMulImpl` comes from `tbe.tik.api.cube.matmul`.
- `MatmulApiCls` is declared in `tbe.tik.api.cube.tik_cube_api` with fields `mat_a,mat_b,m_num,k_num,n_num,l1_bias`.
- `new_stmt_scope(disable_sync=False)` permits compiler automatic synchronization; that does **not** prove a per-exit full-barrier cost or a removable fraction of measured time.
- Disabling automatic sync at a containing scope requires every internal producer/consumer dependency to be managed; external PIPE_ALL barriers are insufficient (R217 demonstrates real L0A/L0B conflicts).

Installed CANN/SDK sources are read-only local references, **not** publicly redistributed in the repo. Their files are excluded from version control; [fingerprints](sdk_reference_hashes216.json) preserve what was read. Only experiment code and API/signature facts are published.

The seeded baseline repeated ≈154.51–154.55 ms at actual Br128/Bc304, kernel_N43776 with synthetic key_limit41792, output SHA `e74e41c74840e5a660513bda508aada3d790531f23b43e5d21d9a616cf465132`. The production-boundary R217 probe instead uses original_pad23/key_limit41769, so its output SHA is not expected to equal that R216 synthetic output.
