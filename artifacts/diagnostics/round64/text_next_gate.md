# Read-only handoff: text encoder next gate after the VAE service experiment

This is a source/receipt audit, **not a new NPU experiment or achieved speedup**.
The current service experiment does not install the experimental text stack.

1. The stable4.096/4.109s figures in
   [Round60 summary](<../round60/summary.json#L176-L195>) are resident50-layer text-only
   TP8 with fixed experimental FP32 Linear/causal math, K/V padding, native BF16
   SwiGLU and RoPE. They exclude processor, vision and offload. They are not full
   production text-encode stage times.
2. There is no closed per-op attribution of that4.1s stack. Controlled deltas:
   [Round59](<../round59/summary.json#L535-L592>) padded eager~5.00s→SwiGLU4.17–4.22s;
   unaccepted Norm fusion2.724s. Same-round
   [RoPE A/B](<../round60/README.md#L146-L163>) is4.218/4.235→4.096/4.109s. Do not turn
   these cross-run deltas into a falsely additive profiler chart.
3. [Round57 native GQA70s/76s dominance](<../round57/README.md#L18-L29>) is an old,
   rejected path, not the current4.1s attribution. The isolated replacement full
   FP32 causal operator was~5.85–5.98ms/layer; multiplying by50 is not a measured
   current-stack attention total.
4. Norm2.724s fails stable final-tensor acceptance by6.318% relativeRMS.
   [Native Qwen order](<../round59/candidate_padded_text_encoder.py#L438-L458>) must
   preserve BF16 residual rounding, FP32 variance/rsqrt, BF16 normalized rounding
   before gamma, then BF16 output. FP16 causal-FA and HF-RoPE variants likewise
   have retained quality counterexamples; no approximate/sparse/INT8 shortcut is
   accepted as lossless.
5. [K/V instability](<../round59/README.md#L79-L104>) was before communication in
   narrow128-output FP32 GEMM. Row padding, determinism and resident operands did
   not fix it; padding to1024 output did in the tested fixtures. Keep that stable
   geometry, not an original-dispatch bitwise promise.100 padded FP32 K/V weights
   are already cached, costing2,097,152,000 bytes/rank; this cannot be sold again as
   a new cache optimization.
6. The most controlled next candidate is **exact FP32 shadow weights for non-K/V
   projections plus one shared FP32 conversion of the identical Q/K/V activation**.
   Keep projection shapes, FP32 causal math, Norm rounding and row-projection
   FP32 all-reduce order unchanged; do not fuse QKV GEMM (that changes dispatch).
   [Existing Q microtest](<../round59/summary.json#L89-L128>) found recast2.931ms vs
   resident FP32 operands0.510ms with a stable Q output hash. This motivates an
   A/B, not a text≤2s promise.
7. Acceptance must freeze/verify actual activations and weights, compare same-input
   layer outputs then all50-layer rank outputs/repeats, and separately account for
   shadow HBM and full-service staging. Reuse
   [K/V version/storage identity guards](<../round59/padded_fp32_projection59.py#L22-L42>)
   as a starting point. A cached resident stack must not be relabelled a co-resident
   full H3 service result.
