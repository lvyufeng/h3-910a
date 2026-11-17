# Round73 — actual vision attention input-cast bottleneck and exact strided kernel

## Complete encoder result

Replacing only vision attention's three BF16→FP32 input conversions/materialization
with an exact strided TBE kernel improves complete resident **encode_ids** from
**6.46375s to5.38407s (1.20053×)**. Native conversion restored measures6.45490s
(-0.137% drift). This is a1.07968s incremental gain and still **not encoder≤2s**.

Three same-call rank maxima,median taken after rank maximum:

- Native6.47631/6.46375/6.43665s.
- Strided cast5.39358/5.37805/5.38407s.
- Native restored6.47082/6.43677/6.45490s.

[Verified summary](<summary.json>), [rank0 proof](<ranks/cast_rank0.json>) and
[corrected complete encoder driver](<cast_attention_fixed_tp8.py>) retain all8
ranks. **96 complete outputs**,including every call's fresh vision/three
DeepStack/real embedding/text inputs/positions/CPU result,match the fixed Round72
reference. First candidates also pass **1,296 actual complete FP32 cast proofs**
and **432 actual complete attention-output proofs** (three operands×54 image/
block calls×8 ranks,and54 outputs×8 ranks).

The task is still the original2,043-token,two48×84-image BF16 encoder,using the
original already-FP16 transport pixels and actual original BF16 weights. The
Round72 partial vision RoPE,Round71 vision Linear mode and Round70 fast text math
remain fixed. No QK GEMM,scale,softmax,PV GEMM,output BF16 rounding,head geometry,
attention masks or model parameters change. Output identity is relative to the
stable experimental stack,**not** independent HF/full multimodal quality,
converted-FP16 production equivalence or official lossless49-step generation.

## Measured bottleneck,not an inferred FP32 ceiling

The independent [inner profile](<profile_summary.json>) transcribes the original
vision attention one operation at a time and uses preallocated/primed events.
[Stage wrapper](<attention_stages73.py>) adds no per-operator synchronization and
releases the unscaled score temporary before softmax,matching the original
expression's lifetime rather than retaining an extra1GiB matrix.

This run compares clean-before/profile/clean-after. Complete medians are
6.46948/6.46414/6.43759s;profile perturbation -0.0825%,after drift -0.493%.
All80 outputs and intermediates match. The median profile call0/rank7 has
complete wall6.46414s,inclusive vision event4.20391s,inclusive attention event
1.78536s. Actual flat attention children:

- Q cast0.28268s,K cast0.27978s,V cast0.38088s: **0.94333s**.
- Their contiguous calls total0.01736s:cast/materialization combined **0.96069s**.
- QK0.21068s,PV0.18701s: **0.39769s**.
- Scale0.10933s,softmax0.22870s,output cast0.08392s.

Flat children total1.78034s;attention-minus-children0.00503s stays unassigned.
The parent vision and attention intervals are **inclusive** and never added to
children. Host-dispatch intervals are separate from device-event intervals.
Events include dependencies/dispatch gaps/stalls and are not pure kernel-occupancy
or roofline figures. Using total attention wall to infer a FP32 GEMM throughput
ceiling would misattribute more than half of this time to math;the actual inner
measurement disproves that inference.

Each profiled call contains **649 events**:one vision parent,54 attention parents
and11 children per attention. The three-operand loop produces six separate
cast/contiguous children,followed by QK/scale/softmax/PV/output-cast. The exact
649-event coverage is validated,not reused from Round71's290-event vision profile.

## Real captured inputs and geometry controls

Selected calls0/17/33/53 correspond to block0/image0,block8/image1,block16/image1,
block26/image1. First-only excluded captures save actual BF16 Q/K/V and original
BF16 attention output,with same-input original-function bit checks and complete
input shape,stride,storage offset/storage bytes,physical format and alignment.
Each fixture is about37.16MB;four total **148,644,724 bytes**,under64MiB/file and
256MiB total limits. No1.04GB score/probability matrix is saved as a fixture.

The [call0](<profile/attention_call00.json>),[call17](<profile/attention_call17.json>),
[call33](<profile/attention_call33.json>) and[call53](<profile/attention_call53.json>)
manifests describe the dense saved copies **and separately the original strided
runtime layout**. Standalone probes reconstruct storage/view stride/offset
explicitly. They do not call a saved contiguous copy the original packed view.

[Geometry probe](<geometry_probe73.py>) and [results](<geometry_summary.json>)
check **180 full matrix proofs**:four fixtures×nine modes×QK/scaled scores/
softmax/PV/BF16 output. Head-padding72→80/128,PV-output padding and query chunk
1008/2016 are mathematically neutral here and all full matrices actually match
bits,but do not produce useful speed improvements in this implementation.
Naive chunks are substantially slower because they repeat full K/V conversions
inside every chunk;this does **not** rule out a future shared-cast chunk strategy.
The no-contiguous variant gives no persuasive gain. These are fixed-order
standalone controls,not integrated complete-encoder speedups or general
mathematical proofs of every padding/layout dispatch.

## Exact strided conversion contract

The [guarded kernel](<vision_cast_guard73.py>) reads one full aligned BF16 token
row:16 adjacent72-wide heads = **1,152 words /2,304 bytes**. The source token pitch
is1,152 for Q/K or3,456 for packed V;image segment offsets remain as recorded.
BF16 words are expanded to FP32 by the proven integer transpose codec. It writes
72-float/288-byte rows per head directly into native contiguous BNSD output
`[1,16,4032,72]`,stride`[4644864,290304,72,1]`.

No FP arithmetic,FP16 intermediary,NaN normalization,head padding or gather is
used. UB is1,280 elements,but only1,152 valid input words are loaded and actual
72-float head rows are written. Source bounds are checked against the last actual
read,not a full unused row pitch. All buffers must be current-device aligned ND,
and unsupported input geometry fails closed rather than being forced contiguous.
The5-pointer launcher uses one input,three unused input aliases and one output,
matching the existing codec ABI.

The [attention factor wrapper](<cast_attention73.py>) changes only the input
conversion and retains the original expressions:

```python
scores = torch.matmul(qf, kf.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
probabilities = torch.softmax(scores, dim=-1)
output = torch.matmul(probabilities, vf).to(q.dtype)
```

All three same-input FP32 tensors,actual output/native layout,input values and
complete BF16 attention output are compared in the excluded first candidate on
all54 calls per rank. These double-computations/copies are off for warm timings.

## Independent original/wrapped/restored controls

A separate [control driver](<attention_control_tp8.py>) and [summary](<control_summary.json>)
compare the genuinely original function against the transcription used by the
inner profile:original6.43340s,wrapped6.42267s,profile6.43426s,restored6.40077s.
Wrapper perturbation is-0.167%,profile versus wrapped+0.180%,restored endpoint
drift-0.507%. All **112 additional complete outputs** (96 warm plus16 excluded
first-call outputs) and intermediate identities match. The excluded wrapped
first call checks **all432 actual attention outputs**
(54×8 ranks),not merely the four saved fixtures. Together with80 initial profile
outputs and96 cast A/B outputs,this round retains **288 complete encoder outputs**.

The original/wrapped phases disable event recording;first transcription proofs
are excluded from their warm medians. This control does not make profile
quantities pure kernel times or eliminate all fixed-order drift. Its small
perturbation also does not justify mixing absolute timings across runs. The
cast's1.2005× complete-stage result uses its own native/restored A/B,not the
control run's lowest native median.

## Preserved failures and finite-only certification

Three distinct failures were investigated,not overwritten or promoted to passes:

1. [Initial singleton-layout failure](<cast_micro_layout_failure.json>) has0
   differing words,but `N=1` native casting preserves an unused singleton stride
   while standard`empty` canonicalizes it. [Diagnosis](<cast_singleton_diagnosis.json>)
   records the rejection. A new run/source uses`empty_strided` only forN=1;the
   actual N=4032 path is canonical contiguous. The strict layout gate remains.
2. [Generic strided-word micro](<cast_micro02_layout_failure.json>) finds NaN
   payload differences under native strided conversion. [NaN diagnosis](<cast_nan_diagnosis.json>)
   preserves examples where raw integer expansion keeps the original payload
   but native canonicalizes it. This is **not** a generic all-word Torch dtype
   replacement. No repair was silently added.
3. [Initial full encoder harness](<cast_attention_tp8.py>) reused the global
   `original_attention` name for vision and accidentally overwrote the text
   wrapper's captured original method. All8 [failed rank receipts](<failed_ab/cast_rank0.json>)
   are retained and rejected by the success aggregator. The new independent
   driver/root renames only that vision binding to`original_vision_attention73`;
   a source regression pins the original text binding exactly once.

[Finite micro evidence](<cast_finite_micro01.json>) deliberately certifies the
finite BF16 domain separately:1/5/61/4032 tokens×1,152/3,456 pitches,native full
bits,integer oracle,shape/stride/physical format and complete source readonly.
The dense61-row corpus covers **65,280 distinct finite BF16 words**. Nonfinite
words are excluded from this finite fixture generation,not masked in model
outputs or repaired in the kernel. Actual model input/output proofs are finite
and all match;the generic NaN counterexample remains unaccepted.

Standalone4032-token cast medians areabout0.477/0.487ms against3.384/6.774ms
native for QK/V pitches. Those isolated times are **not substituted** for the
model's measured0.9607s category or the complete1.07968s A/B saving. The micro's
timeout wrapper ran on owneddevice1 while the independent geometry probe ran
on device0;their shared-host contention means these are diagnostic observations,
not uncontended canonical throughput or full-stage samples. No claim that
the entire difference is uniquely pure cast kernel time is made.

## Timing,precision and memory boundary

The complete timer includes real embedding,fresh vision/three DeepStack,
positions/mask,text50,the original`encode_ids` CPU result and ending drain.
Start barrier/drain are outside;extra identity copies/hashes are after stop.
No capture hook or event recording runs in the clean cast A/B. The original
entry already uses`@torch.inference_mode()`;precision flags remain actual
matmul-HF32=false/conv-HF32=true,and all actual input/model bytes are unchanged.

Allocated HBM snapshots remain23,888,436,736 bytes/rank. Reserved memory grows
27,965,521,920→29,873,930,240 during first candidate proof/compilation and stays
reserved after restoration. This is not peak usage,proof that the new kernel
reduces memory,or evidence of DiT/VAE co-residency. Existing11.1084GiB text shadow,
2.20048GiB vision shadow and padded K/V remain fixed. No activation output is
cached across calls;only compiled object handles are cached.

## Final verification and unfinished gates

[Final fingerprints](<final_verification.json>) verify23 unchanged production/model/
dedicated-module sources,36 executed diagnostic/dependency files(including failed
and corrected sources),five actual reference/attention-fixture files and eight
compiled strided-cast object/metadata pairs. All match. Tensor fixture blobs remain
retained remotely;local manifests and returned verification receipts are not
misrepresented as an independent local rehash of absent blobs.

[Cleanup receipt](<cleanup_receipt.json>) records all8 NPUs idle,zero accelerator
processes and the owned lease released. No production source or replacement
service was modified. Actual numerical evidence lives in the micro,geometry,
profile,control and fullencoder rank receipts,not in CPU mock substitutes.
[Regression receipt](<test_receipt.json>) records16 new tests plus159 prior tests,
**175 passing** without skipped actual-data gates. It pins complete timer/ABI,
fixed text/vision bindings,cast/attention raw bits and layouts,finite-domain
coverage,geometry full matrices,inclusive profile categories,the retained
failures and independent original/wrapped/restored/source/fixture controls.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round73 -p 'test_*.py' -v
python3 artifacts/diagnostics/round73/summarize_cast73.py
python3 artifacts/diagnostics/round73/summarize_profile73.py
python3 artifacts/diagnostics/round73/summarize_controls73.py
python3 artifacts/diagnostics/round73/summarize_geometry73.py
```

Original vision attention/RoPE/Linear wrappers restore and shadows clear at
exit. Production integration,loader/offload hooks,arbitrary NaN-bearing inputs
and full service residency remain out of scope. The overall objective remains
active for full encoder≤2s,independent HF multimodal quality,strict DiT<15s,
full A/V identity and task/precision/warm-protocol matched49-step comparison.
