# Round 51 — local VAE padding-layout compatibility repair

## Controlled padding equivalence

The original 910A packed-input `reflection_pad3d` failure was reproduced in
separate FP16 and FP32 processes. Native packed input is rejected. Explicit
conversion to either ND (2) or NCDHW (30) before the same reflection pad succeeds.
Across **12 shape/dtype/layout cases**, both converted input values and padded
output values match CPU exactly, with dtype unchanged. The small-tensor
convert-plus-pad medians range **0.1525–0.1732 ms**. This is not a whole-VAE speedup.

See [raw probe records](<pad_probes/nd_float32.json>),
[probe source](<pad_format_probe.py>), and [summary](<padding_summary.json>).
The first harness tried to read an unavailable runtime-config attribute; it was
corrected to a read-only optional query before any measurements. No global
internal-format, power, clock, model precision, or weight setting was changed.

## Two actual padding entry points

The first patch locally converts packed input to NCDHW only before reflection
padding in `Downsample3D`. CPU behavior, non-reflection modes and the separate
spatial-exchange branch remain unchanged. The helper and real forward wiring are
covered by **7 CPU/NPU tests**.

Loading the real encoder alone then found another entry point, retained in the
[first encoder failure](<vae_encode01/result.json>): a residual `1×1×1 Conv3D` with
zero padding still delegates to `nn.Conv3d`, which calls `F.pad(..., reflect)`
with six zeros. The numerically empty padding call rejects the internal layout.
This failed before an image latent was produced; it is not a valid encode time.

The second repair:

- removes only that zero-padding no-op and calls the identical `F.conv3d`;
- converts to base NCDHW before explicit reflection padding in BaseConv3d and
  spatial-parallel boundary padding;
- retains stride, dilation, weights, bias, dtype and existing temporal padding.

**6 further CPU/NPU tests pass**: CPU zero/padded convolution behavior is bit-exact
to the original classes, reflection padding values on NPU are exact to CPU, and
the packed-input zero-padding shortcut runs and matches CPU convolution within
the stated backend tolerance. This last convolution check is not a bitwise
CPU/NPU arithmetic-equivalence claim.

[Tested VAE candidate](<candidate_vae_cnn.py>),
[Conv3D candidate](<candidate_conv.py>), [padding tests](<test_padding.py>), and
[convolution tests](<test_conv_padding.py>) are provided. The
[combined portable patch](<../../design/patches/vae_npu_reflection_pad.patch>) is
applied to fresh copies of both originals and verified byte-for-byte against
the tested candidates.

## Deployment and true encoder verification

Canonical model source targets were resolved before editing: the fp16/4step model
links point to the original model's VAE code; no weights were altered. Exclusive
backups preserve the old files, and all target source hashes are checked before
writes. Known cached source copies are patched as well, but the next HF dynamic
module may use a fresh cache hash. The real encoder probe checks which actual
class files loaded and verifies that both tested helpers are present, rather
than assuming the old cache directory is still the active one.

[Downsample deployment](<deployment.json>) and
[Conv3D deployment](<conv_deployment.json>) preserve the exact target/hash receipts.
Scope is local VAE padding only; the 30-core DiT FA and round-50 BF16 vision
compatibility paths remain unchanged.

The second standalone **real-weight** encoder probe succeeds at both sizes:

| Input | Output rows | Observed call time | Result |
|---|---|---:|---|
| 64×64 | `[4,96]` FP32 on CPU | 98.269 s | finite; repeat is bit-exact |
| **1344×768** | **`[1008,96]` FP32 on CPU** | **187.956 s** | **finite and correct shape** |

These first observed calls include cold-path effects and use a single NPU;
**they are not steady-state times, VAE decode times, or speedup claims**. CPU
weight loading (41.827 s) is measured separately. The existing FP32 encoder,
native keyframe RNG seed and CPU normalization/patchification are retained.
Encoder-only loading removes unused decode modules, not active encoding work.

The actual fresh dynamic-module cache loaded both tested sources. The recorded
trace contains **1020 padding calls**, including **120 packed(32)→base(30)** calls.
Every packed reflection padding output is in format 30. Evidence:
[full raw encode record](<vae_encode02/result.json>),
[compact encode summary](<vae_encode_summary.json>), and
[reproducer](<vae_encode_probe.py>). The **13 passing regressions** are recorded in
[the test receipt](<test_receipt.json>).

## Full real-workload result: entered DiT, but stopped before video output

The complete first/last-image 768p/5s form passes the previous padding failures.
All eight ranks report **507.71 s** for encode/conditioning preparation, then
complete the first DiT denoiser evaluation. Its cold progress bar reports
**351.31 s** for iteration 1; model timers are about 296.7–296.9 s plus separate
prep/update reports. The test manager stops the still-running request at its
**900 s deadline during the second step**, before denoise or video completion.
This is a harness timeout, not a new operator-error verdict.

The [request receipt](<e2e01/result.json>) and
[partial summary](<e2e_partial_summary.json>) distinguish these observations from
steady-state performance. Encode/conditioning is not an isolated text-encoder
measurement. Individual late blocks (~0.54 s) are not a complete steady step;
pre/post/offload overlap must be measured, not inferred by subtraction. No 15 s
DiT, 2 s text encode, 20 s VAE decode or official lossless success is claimed.

## A new measured attention opportunity at the actual FL2VA shape

The real request has **N=41792**, H=7, D=128 on every rank. The existing exact-divisor
heuristic selects **Br=64/Bc=64**, not the 128×304 tile used at the old 63232 shape.
The runtime core budget is still correctly 30.

A controlled prototype compares the same valid Q/K/V inputs to zero-padding the
sequence to **43776 (=18×lcm(128,304))** so Br=128/Bc=304 can be used. The existing
LSE correction removes the contribution of the **1984 known-zero keys** from the
softmax denominator; padded values contribute zero to the numerator. Both builds
use the same deployed 30-core wrapper and no production source changes.

| Wrapper call, same valid input | First order median | Reversed-order median |
|---|---:|---:|
| 41792, 64×64 tile | 421.375 ms | 421.375 ms |
| 43776, 128×304 tile + LSE pad correction | **154.202 ms** | **154.211 ms** |

This is **2.7325× faster / 63.40% less latency** at the wrapper-call level. Padded
buffers are prepared before timing; a production internal-padding integration
must account for its extra input copies and allocations/reuse before claiming
that end-to-end gain.

Whole-valid-output pair relative RMS difference is at most **0.03622%** on two
input amplitudes. The maximum sampled relative RMS error to the FP32 full-key
reference is **0.04097%**; sampled max absolute error is **7.221e-6**. This reference
checks eight query rows of one head over all valid keys. Outputs are finite but
**not bit-exact** across tilings, and these checks are not a whole-model quality
or official lossless result.

[Same-input prototype](<fa_padding_ab.py>) and
[raw A/B/numerical evidence](<fa_padding_ab_result.json>) are provided. The tuned
padding strategy is **not deployed** yet. The next round should integrate this
with shape/cache/zero-key safeguards and compare real model outputs and timings.
The tested VAE padding repairs remain deployed, originals backed up. All managed
processes are stopped, eight cards are verified idle, and the fleet lease is
released at round completion.
