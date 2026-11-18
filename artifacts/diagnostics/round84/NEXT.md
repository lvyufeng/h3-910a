# Next — stop treating FP32 row partials as the remaining precision fix

## Verified R84 facts to retain

- Actual official text checkpoint has intermediate size **25600**: TP8 MLP
  down projection is K3200/N5120; attention output projection is K1024/N5120.
  The old K3456 experiment is a valid generic GEMM shape, **not this model's
  actual MLP geometry**. Use [checkpoint shape receipt](<actual_shape_receipt_fixed84.json>).
- [Corrected private row adapter](<row_fp32_adapter_fixed84.py>) now really ran
  all50 layers on all8 ranks. Half operands → FP32 GM partial → FP32 HCCL →
  one Half output,804 calls/rank with both collective dtypes recorded.
- Same CPU FP16 conditions: selected50 relative RMS **2.4757%→2.1902%**.
  Native NPU vision conditions: **7.4201%→7.5777%** (slightly worse).
  One actual CPU-input decoder block: 1.457e-4→1.034e-4. All repeats bit-stable,
  parameter/input hashes unchanged, original native output fully recovered.
- Genuine warmed private `encode_ids` with FP32 rows: **2.433648s**, not <2s.
  Half weight-layout cache costs **2,162,688,000B/rank**, scratch101,187,712B.
  Do not make this the production default or claim this is a speedup.
- CPUFP16 gold remains unchanged; no old BF16 result cast, FP32 shadow weights,
  new numerical tolerance, or original-weight/model-code change is allowed.

## Next concrete precision experiment: actual attention operands

The R79 norm already follows the CPU Qwen3VL order: Float reduction, Half
normalised value, Half gamma product; residual addition is Half. Do not change
its two-rounding contract to chase precision. Remaining attention arithmetic
is a direct next test, not an attribution proved by R84.

1. Capture actual Half Q/K/V immediately before the scoped text SDPA helper on
   layer0 with exact CPUFP16 conditions. Keep original Half parameters, KV zero
   padding and rotary math unchanged; compare the operator against **actual CPU
   FP16 `scaled_dot_product_attention` with MATH backend** on exactly the same
   captured operands. Freeze and hash operands; include causal mask and scale.
2. Separately compare actual score matmul, softmax and P@V to their CPU math
   intermediates to identify the first large discrepancy. Existing R80 live
   controls showed native Float matmul may still have reduced internal precision
   and HF32 toggles did not repair it; don't assume a Float API means true FP32
   accumulation just because its returned dtype is Float.
3. If score/PV is confirmed, use the verified Half→FP32 L0C kernel primitive,
   but **batch heads** instead of many tiny per-head Python launches. Actual TP
   text attention has 8 local heads, head dim128, sequence2043, query chunks512
   (last507), key rows padded only with zeros to2048 and discarded/masked.
   - Q/K are original Half: accurate Half×Half→FP32 score output, same scale
     and causal softmax in Float. No early Half score/probability cast.
   - Probabilities are CPU-reference-style Float, V original Half. Test a
     Half-high + scaled-Half-residual decomposition of P (power-of-two scale,
     e.g.1024 to avoid losing tiny residuals to Half subnormal rounding), with
     both matrix products accumulated/output in FP32 and their weighted sum
     formed in FP32 before the single Half attention output.
   - Keep all intermediate partials Float; native Half-output matmul followed
     by `.float()` is not a substitute. Verify decomposition/underflow behaviour
     on actual P before using it, not by random-only benchmarks or a made-up
     FP32 full-graph teacher.
4. New batched TIK source, new kernel names and new write-once roots; keep the
   executed R83 source unchanged. A batch dimension must select the correct
   per-head K/V matrix; flattening heads across a shared weight matrix is wrong.
   Head batching is especially necessary for PV N128, which otherwise gives
   only four 128×128 output tiles per query chunk and underuses 30 cores.
5. Verify one actual operator and raw decoder block first, then the complete50
   CPU-condition and native-condition A/B, native recovery, repeat/hash gates,
   stock full encode time and extra buffers. Neither finite nor repeat equality
   is a full precision pass.

## Independent next leaf experiment: real vision biased linears

R83's portable FP32 UB bias + one Half cast reduced random bias-Linear error
104× at a micro size and26× at H3 KV size. R79's vision Half branch still calls
native biased Half Linear, so the extra Half GEMM-output+bias rounding remains.
That is a justified candidate for **measurement**, not yet a full vision fix.

- Read actual checkpoint geometry first: vision hidden1152, MLP4304, merged
  hidden4608/out5120, M8064 beforemerge/M2016 aftermerge. Both QKV/proj and MLP
  linears are biased, as are merger linears. No more unverified width assumptions.
- Freeze actual first-block input and original Half weights/bias; compare native
  and custom directly to CPU Half `F.linear` with the same operands.
- Use [verified portable bias kernel](<../round83/blocked_bias_epilogue83.py>),
  not unsupported TIK hardware bias. For MLP4304, zero-only K/N padding to4352
  and exact cropping need new tested host contracts; check untouched prefix,
  zero-only padding and output-row bias semantics. No truncation or rounding of
  original values, no FP32 weight shadow.
- Then validate all27vision blocks/final vision/3DeepStack against the unchanged
  CPUvision payload and complete text conditioning. Keep separate row/vision
  switches; R84 native-condition regression means enabling both blindly would
  obscure which branch actually helped.

## Performance and pipeline goal still open

Accuracy work must be followed by fuse/persist activation-layout and epilogue
operations without changing the CPU Half rounding contract. Cache/setup work
must not disappear from the service memory or end-to-end lifecycle accounting.

Earlier full DiT49 warm evaluations mean14.657s/all<15s and VAEservice13.52–13.62s
retain their original qualifications. Whole encoder <2s, original CPUFP16 graph
agreement, real encoder/DiT co-residency, and official task-matched A2/A3 pipeline
comparison are still active. Official150.06s(A2)/114.19s(A3) rows are T2VA/49;
our FL2VA/FP16 experiment is not an official equivalent just because it is768p.

All R84 jobs/required fresh-fork result collected, production23/API13/CPUgold/
launcher verified unchanged, relevant workers absent and ownedts217 released.
Other sessions later claimed CUDA133/134; leave their leases alone. Live-claim
owned again before NPU work, release unused grants, no CI/dormant opt-in. All new
subagents must be **fresh forks**, never continue inactive old agents. Goal16
remains active/armed for the next round; task completion here is not whole-goal
completion.
