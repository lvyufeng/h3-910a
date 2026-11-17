# Round79 — coherent FP16 vision runs on910A; numerical alignment still pending

## Actual device result

The [scoped vision probe](<vision_probe79.py>) executed the
[generated diagnostic candidate](<candidate/candidate_encoder79.py>) on one910A
NPU with original dense weights converted toFP16. It used the independently
recomputed CPU FP16 vision tensors,**not an oldBF16 output cast**.

[Validated summary](<summary79.json>) and [actual receipt](<vision.json>) show:

- all27 vision blocks,final merged features and3 DeepStack outputs are finite;
- all351 parameter tensors areFP16 (1,190,533,600 bytes),and every converted
  vision parameter hash matches the CPU reference conversion;
- three NPU repetitions are bit-identical to the first NPU output;
- warmed resident **vision-only** times are1.434726/1.436732/1.474425s,
  median **1.436732s**;
- relativeRMS difference vsCPU FP16 is **1.456805%** for final vision features,
  and **0.281997/0.565813/0.908825%** for the three DeepStack outputs;
- final vision maximum absolute difference is0.710938.

The excluded first diagnostic call took146.599s and includes cold/operator
initialization plus per-layer proof copies/checks. It is not a warmed number.
CPU reference execution was concurrent;this is not a cross-round controlled
speed A/B. In particular,**1.44s vision is not complete encoder<=2s** and cannot
be compared directly to the earlier5.37s complete BF16-compatible encoder.

The probe selected no numerical pass threshold after seeing the data. We do
**not** declare complete FP16 precision accepted:the1.46% feature difference
still needs same-input operator controls,and TP8 text row-GEMM rounding remains
separate. The user's CPUFP16 criterion is honored;BF16 agreement is not restored
as a gate.

## Reference provenance

[Official-source verification](<CPU_source_verification79.json>) checks all12
used local checkpoint shards against the14-file pinned official LFSSHA256
manifest at [MiniMaxAI/MiniMax-H3@0919fb68](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/text_encoder).
The real reference has903 FP16 parameters,original FP32 rotary buffers and the
same2043-token,two-image fixture. Its first vision features are complete. The
[archived CPU progress](<CPU_progress02.json>) observed48 completed language
layers;the latest [status snapshot](<CPU_progress03.json>) tracks the ongoing
full repeat. Reference wall time is not NPU performance.

## Candidate and scope corrections

The [deterministic generator](<candidate/build_candidate79.py>) applies12
explicit edits to the verified production architecture. The
[manifest](<candidate/build_manifest79.json>) pins candidateSHA`2141b29a…`.
It removes BF16 pixel/output round trips,ports originalHF FP16 RMSNorm rounding
and textRoPE,and preserves FP32 vision position interpolation.

[Source-scope correction](<source_scope_correction79.json>) distinguishes the
actual production architecture from the earlier separateR65 shadow-cached
experimental encoder. The latter—not the production source—contains the
FP32-shadow/padded helper flags. A FP32 all-reduce also cannot recover precision
already rounded out of a FP16 partial GEMM;the single-device vision probe does
not fix or measure that TP8 text issue.

[Post-probe verification](<post_probe_verification79.json>) confirms all23
production and13 installed API/metadata source files remain unchanged,and no
NPU probe process remains. The independent CPU reference process remains alive;
the ownedhost lease is retained for it rather than declaring the machine free.
The required fresh-fork implementation was collected and reviewed before the
probe;no inactive old teammate was resumed.

## Network and archive handling

A live12s probe briefly reported a SSH timeout. The original CPU process was
not replayed or considered stopped. The optional83MB local reference archive
was postponed;the existing remote write-once tensor was used for NPU comparison.
A later30s live probe returnedFREE/noJenkins and the staged source was verified
before the NPU run. The partial local archive remains explicitly unverified and
is excluded by the [local ignore rule](<.gitignore>);it is not published as gold.

## Next work

1. Finish and verify the original full CPUFP16 repeat,without launching a duplicate.
2. Run same-input vision Linear/LayerNorm/Conv/attention controls to locate the
   accumulated1.46% difference;do not silently label it “justFP16 rounding”.
3. Validate completeFP16 TP8 text/vision against the same CPU reference,including
   pre-finalnorm50 boundary and partial-GEMM/collective order.
4. Only then integrate/fuse the candidate and measure the complete warmed
   encoder target2s. DiT/VAE stage qualifications and officialT2VA comparison
   remain separate.

[State](<STATE.json>) preserves the actual completed probe and the remaining
CPU/NPU acceptance work. The whole optimization goal remains active.
