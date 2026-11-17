# Round76 — actual 49-step videos: speed measured, visual quality rejected

**910A has no native BF16 computation.** The new experimental encoder stores
BF16 tensors but computes through FP32/integer codecs. The actual video engine
uses the native, unfused **FP16 DiT runtime**. This is not an original-BF16
lossless claim.

## Generated results

Two real requests used the same red-ball prompt, first/last anchor images,
1344×768, 5 seconds, 24fps, seed1234, FL2VA and **49 steps**, with no FlashGen or
other LoRA. Both outputs contain **124 decoded frames / 5.1667 seconds**.

- [Optimized-encoder conditioning video](<optimized.mp4>) · [key frames](<optimized_contactsheet.png>)
- [Independent HF encoder-conditioning control](<HF_reference_conditioning.mp4>) · [key frames](<HF_reference_conditioning_contactsheet.png>)

![Actual optimized-conditioning key frames](<optimized_contactsheet.png>)

**Both fail visual quality.** The first/last red-ball anchors are approximately
preserved and the ball moves/bounces, but the middle contains a prominent,
unprompted logo/lettering, a yellow radial glow and visible ball ghosting/colour
fringing. Initial motion goes left before moving right and overshoots the final
anchor. The HF encoder-condition control changes the logo's glyph detail but
**does not remove these faults**. Therefore correcting this encoder path alone
is insufficient; the sole cause is **not established**.

The [quality summary](<quality_summary.json>) combines the actual
[124-frame scan](<video_scan.json>) and visual inspection of **11 sampled time
points per video** (0/12/24/36/48/62/74/86/98/110/123), plus original-resolution
optimized frames0/12/24. It does not claim manual inspection of every frame.
Red-colour centroid and frame-MAE are diagnostics, not verified object tracking,
optical flow or a perceptual quality score. Endpoint PSNR compares reconstructed
anchors only, with original RGB PNGs reported separately from the lossy MP4.

## Actual performance — cold and warm kept separate

[Final generation receipt](<generation_result.json>), [sealed complete log](<generation_complete.log>)
and [synchronized timing summary](<generation_summary.json>) are authoritative.
Full-step numbers take the **maximum of eight ranks for the same step**, then
average the requested steps; they are not block-stack timing or max-of-sums.

| Observation | Optimized conditioning, first/cold request | HF conditioning, second/warmed request |
|---|---:|---:|
| Request wall, **excluding external encoder** | **958.526s** | **735.689s** |
| First full step | 62.272s | 14.702s |
| Second full step | 20.041s | 14.704s |
| Steps2–49 mean of same-step rank-max | 14.775s | 14.658s |
| Steps3–49 mean / maximum | 14.663 / 14.797s | 14.657 / 14.733s |
| All49 steps strictly below15s | **47/49 — no** | **49/49 — yes** |
| Local conditioning/media `encode` reporter | 56.69s | 1.91s |
| `denoise` reporter | 776.40s | 719.36s |
| `decode` reporter, rank range | **122.06–128.27s** | **13.52–13.62s** |

Startup **173.546s** is excluded. The two rows differ in conditioning **and**
thermal order, so their wall-time difference is **not** a controlled encoder
speedup A/B. Do not sum per-stage rank maxima to reconstruct request wall.

The decode reporter includes denoise-cache release/residency, video VAE decode,
crop and audio VAE decode/audio D2H. It ends **before** video uint8 quantization,
video D2H, worker/engine transfer and our MP4 export. Thus the warm13.62s reporter
is not the earlier Round64 full-service13.81s median, nor proof of cold decode
under20s. The cold request clearly misses20s.

## Encoder and precision boundary

[Fresh complete encoder proof](<encoder_summary.json>) recomputed real embedding,
fresh vision, M-RoPE, DeepStack and all50 text layers on eight owned NPUs before
generation. Three same-call rank-max samples are5.37226/5.35007/5.37166s:
**median5.37166s**, still above2s. All32 first/warm complete outputs and recorded
intermediate identities match the fixed experimental output.

Generation then uses the supported **external text-conditioning** payload with
`text_encoder=False` and live local VAE media encoding. The encoder is not loaded
inside this engine, so its compute/load/offload cost is **not inside the above
request wall**. No invented additive live-encoder E2E result is reported.

The [fixed generator](<generate_optimized_fixed76.py>) validates the rebuilt
input IDs, FP16-transport processor pixels/grids and both complete payloads
before loading. The fresh BF16-storage output SHA`67ef196e…` becomes consumed
FP16 SHA`a4642a05…`. The independent original-HF selected50 pre-finalnorm hidden
is also cast to FP16, exactly as the DiT wire requires. Its **40.9511% relative
RMS discrepancy remains**; that is an independent hidden gate, not a video score.
The HF control is an encoder reference on the **same FP16 DiT/VAE**, not an
independently validated DiT or original-BF16 video oracle.

## Provenance, failures and cleanup

- The initial offline call failed on CLI-only `usp`/`ring` names rather than API
  `ulysses_degree`/`ring_degree`. [Failure](<failed01/generation_result.json>) and
  [log](<failed01/generation.log>) remain unchanged; no video was generated there.
- A later SSH client timeout did **not** stop the remote generation. The original
  owned process was reconciled rather than replayed; both requests completed and
  the original VAE was restored. [Restoration](<restoration.json>) and
  [final verification](<final_verification.json>) prove all original workers gone.
- The interrupted archive is **not accepted as a complete transfer**. Five
  complete PNG members were individually recovered and validated against their
  original remote hashes in [recovery proof](<transfer_recovery.json>). Both MP4s
  were separately downloaded and checked against their generation-receipt SHA.
- The interrupted final-log prefix is retained. Exact byte-range recovery plus
  the complete source hash in [log metadata](<final_log_metadata.json>) validates
  the sealed log used for timing. A partial log is not used as the final result.
- Local Homebrew ffprobe failed on missing libtiff. [Failed local scan](<optimized_scan.json>)
  remains; no global dependency/symlink changes were made. The existing remote
  PyAV performed the actual complete scan instead.
- Final CPU rehashes match **23 production sources, 13 API/metadata files,
  31 executed diagnostic sources, four actual conditioning/reference fixtures
  and all13 native unfused FP16 checkpoint files (66.28GB)**. Decoder opt-in was
  instance-local/default-off; production encoder/pipeline were not modified.
- The owned ts-217 lease was released after all remote work and verification.
  [State](<STATE.json>) records the final roots and remaining objective.

Both deliverable MP4s are **video-only**. Audio was decoded (the receipt records
stereo165600 samples at32kHz), but its waveform was not exported or reviewed;
there is no audio-quality or whole-A/V identity claim. Round64's rejected whole
A/V gate is not waived.

## What remains open

The warm same-runtime DiT measurement is below15s for all49 steps, but the video
quality gate, original-HF hidden agreement, live full encoder≤2s, cold decode≤20s
and the whole official-lossless objective remain open. Official A2/A3 numbers
150.06/114.19s are warmed **T2VA**, whereas these are FL2VA with a non-native-BF16
FP16 runtime; they are not directly comparable.

The next useful work is to locate the shared unprompted middle-frame artifacts
across model/conditioning/sampling, DiT attention and VAE paths using independent
stage references—not to hide them behind the faster timings or call this
lossless. **15 Round76 tests and 262 relevant Round57–76 contract tests passed**;
[regression summary](<regression_summary.json>) and [collected suite output](<targeted_regression_result.json>)
record the scoped coverage. The broader old test attempt's missing-Torch
environment failure is retained in [legacy attempt](<regression_result.json>),
not silently reported as a pass.
