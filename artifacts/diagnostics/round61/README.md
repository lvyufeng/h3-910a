# Round61: actual FP16 indexed AdaLN and gated-residual kernels on910A

## Acceptance scope

These are self-developed TIK/TBE **FP16-GM** kernels compiled for **Ascend910A**,
not A2/910B. The reference is the current eager FP16 H3 DiT implementation:
`1+scale`, normalized output, multiplication, and residual addition retain their
intermediate FP16 roundings. This is **not** a claim of equivalence to original
BF16 H3 or the official Triton FP32-intermediate lossless recipe.

The kernels remain diagnostic-only; no production fusion dispatch was installed.
The capture-only transformer modification was reverted byte-for-byte. See
[restoration receipt](<capture_restoration.json>) and
[final source verification](<../round62/final_source_verification.json>).

## Actual workload and input contracts

A single-interval, instrumented FL2VA request uses the same two1344×768 first/last
images, prompt,5s/24fps/seed1234 as the49-step workload. It supplies **actual local
5224×5376 FP16 DiT rows**, two timestep embeddings and six modality-indexed
parameter rows. Captures cover block0/24/49 and the final affine. All rows fit the
6144-row capture bound; raw tensors stay on the leased host under
`/tmp/h3_round61_adaln01/capture`, while the
[capture metadata](<capture_metadata/rms_norm_indexed_scale_shift_000.json>) and
[instrumented request receipt](<e2e_capture/result.json>) are retained here.
Instrumentation timing and one-step video are not quality or clean speed proof.

Actual projection returns non-contiguous chunk views: block row stride32256
(6×5376); final affine row stride10752 (2×5376). The direct point builder loads
only5376 elements from each offset-adjusted table pointer; stride enters both
binary and Python cache identity. Complete block replay in
[round62](<../round62/README.md>) tests the **real chunk views**, not only dense
copies or reconstructed strides.

## Measured operators

[Initial attempt](<attempt01/adaln_validation.json>) retains the slower FP32-UB
point implementation and per-call CPU index checking: gate wrapper~0.80ms versus
~0.36ms eager. That version was not a speedup.

[Prepared-snapshot attempt](<adaln_validation02.json>) verifies all10 actual cases.
Point kernels cache tiny FP16 parameter tables inUB and keep vector operations
separate to preserve rounding. Indexed metadata is copied once into a validated,
owned int32 snapshot, padded to a32-byte DMA block. Ordinary tensor calls still
check mutable index values every time; the faster snapshot API is explicit.

With actual chunk strides, [final point proof](<point_stride_validation03.json>)
measures the complete prepared-metadata wrapper, including fresh output allocation:

- Final affine: eager0.727ms including its input clone → point0.261ms. These are
  functional comparisons; a production in-place adapter must add its copy-back.
- Gated residual at blocks0/24/49: eager0.359/0.363/0.362ms → point0.262/0.248/0.248ms.
- Every output word matches actual captured output and same-input NPU eager.
- N1/7/8/9/129 remainder cases with six table values and nonzero chunk offsets
  also match; source inputs remain unchanged.

Norm-containing variants are **not accepted as lossless**:

- RMSNorm+affine: eager1.578–1.581ms → fused0.743–0.746ms; relativeRMS differences
  roughly5.90–6.48e-6 versus actual eager output.
- Gate+RMSNorm+affine: eager1.874–1.886ms → fused1.020–1.025ms; residual output exact,
  normalized/modulated output differs roughly6.48–7.45e-6.
- FP32 reduction/rsqrt ordering is changed. Exact padding or point operations do
  not certify this norm. [Full block rejection](<../round62/README.md>) shows those
  small local differences amplify after actual attention/MLP.

These isolated operator ratios are **not** whole DiT-step or encoder/decoder gains.

## Safety and finite arithmetic checks

[Final guard/sweep receipt](<point_guard_validation04.json>) verifies:

- Negative/out-of-range indices, snapshot mutation, unvalidated stream changes,
  wrong current-device context, unaligned base pointers and table row strides
  are rejected before custom launch.
- Gradient-enabled gate/branch/norm-weight operands are rejected, even if `x`
  itself has no gradient; kernels are inference-only.
- `x.copy_(launch('mod',...))` returns the original tensor, retains its pointer,
  and updates aliases identically to the reference's in-place final affine.
- All63488 finite FP16 bit patterns across33 tested gate/scale/shift scenarios
  are raw-bit exact to **actual NPU eager**, including signed zeros and extreme
  finite factors:2,095,104 tested scalar outputs. This preserves the NPU arithmetic
  contract, not ideal IEEE CPU behavior. NaN/Inf payloads and arbitrary norm
  overflow/epsilon behavior are not certified.

The isolated [C launcher](<vec61_launcher.c>) leaves the shared production
five-pointer FA launcher untouched. It rejects a65th registration before calling
runtime APIs; a portable fake-runtime C regression verifies that boundary without
claiming64 real hardware registrations. Concurrent registration and general
cross-stream service use remain untested. Assertions are diagnostic guards, not
hardened production API validation.

## Reproduce and evidence versions

Final [AdaLN wrapper](<adaln_vector61.py>) and [point builder](<point_vector61.py>)
are source-pinned by the round62 receipt. Earlier attempts are retained rather
than overwritten. Norm timings above belong to the preserved pre-stride attempt02;
final source is additionally exercised in actual complete block replay.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round61 -p 'test_*.py' -v
# On ts-217, in the existing environment and under a fresh owned lease:
source /home/mseco/Ascend/cann-9.2.0/set_env.sh
/home/mseco/vllm910a/bin/python /tmp/h3_round61_adaln01/point_stride_validation.py
/home/mseco/vllm910a/bin/python /tmp/h3_round61_adaln01/point_guard_validation.py
```

Reproducers refuse existing receipt paths; use a new exact path for a new attempt.
Eight portable tests cover stride spans, separate FP16 operations, padded metadata,
cache/core identity, guards and isolated C registration capacity. Raw hardware
proofs, not structural tests alone, support the numerical claims.

## Goal status

AdaLN affine/gated residual kernels now have actual FP16 operator and whole-block
identity proof. Full Norm fusion remains rejected. Text≤2s, VAE≤20s, sustained
quality-accepted native49-step FL2VA and an official-task/precision/warm-protocol
matched comparison remain outstanding. No new49-step video or encoder/VAE speedup
was produced in these rounds.
