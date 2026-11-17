# Round74 — complete independent original-BF16 HF multimodal reference

## Result: the independent reference is complete;the quality gate is not passed

For the actual same2,043-token,two48×84-image input and original BF16 checkpoint,
installed HF independently executes **vision→real embedding→M-RoPE→masked visual
features→three DeepStack→50 decoder layers**. The original HF finalnorm still
executes;the comparison captures its input,the unnormalized hidden state H3
actually selects. No experimental visual feature feeds the HF computation and
no FP32-linear oracle adapter is used.

Two unchanged-BF16 HF language executions match **all50 block-output raw SHA
values and both complete pre/post-finalnorm tensors**. The newly computed full
HF vision merged/three DeepStack/positions also match retained independently
executed Round54 HF reference tensors. This completes the previously missing
original-BF1650-layer reference;it does not retroactively convert the older
2400s timeout into a successful experiment.

[Verified summary](<summary.json>) reports current fast experimental encoder
versus this independent reference:

- Selected50 full hidden: **40.9510924% relative RMS**,maxabs **15,140**.
-27 text positions: **0.1680393% relative RMS**,maxabs3.
-2,016 visual positions: **43.8680155% relative RMS**,maxabs15,140.
- Both sides are finite. **Hidden identity and official-quality gates remain
  false**. Stable experimental identity from Round73 is a different acceptance.

The numbers above include **all10,460,160 elements**. No outlier/channel/token is
removed,masked or repaired to obtain a smaller headline. Relative RMS is the
square root of total squared error divided by total squared reference values;
full/text/visual use their own complete denominators,not an average of chunk
relative RMS or a sum of modal percentages.

## Error concentration and unresolved attribution

[Modal/error-concentration receipt](<modal_comparison01.json>) computes full-tensor
error sums before reporting top16 channels/tokens:

- Visual positions contribute **99.9997835%** of total squared error.
- Top16 channels contribute **97.6768096%**;top16 tokens **98.1211010%**.
- All top16 error tokens are visual. Largest error-channel IDs include731 and2994.
- HF and experimental absolute maxima are19,712 and19,584,respectively.

These are producer-derived diagnostic statistics from the hash-pinned
[comparison source](<modal_comparison74.py>);the absent remote tensor blobs are
not claimed independently locally rehashed or reanalysed. Concentration is not
permission to exclude those elements:the unnormalized hidden is what H3 consumes.

Selected raw block-output comparisons,**before model-loop DeepStack injection
at0/1/2** on both implementations:

- Layer0:full5.091%,text0.258%,visual5.094%,maxabs3.75.
- Layer1:full4.044%,text0.291%,visual4.047%,maxabs5.5.
- Layer2:full3.307%,text0.320%,visual3.309%,maxabs6.25.
- Layer9:full1.346%,text0.019%,visual2.815%,maxabs16.
- Layer24:full1.834%,text0.045%,visual4.324%,maxabs12.5.
- Layer49:full40.951%,text0.168%,visual43.868%,maxabs15,140.

The large change is **somewhere between selected layers24 and49**,not proved to
occur only inside layer49. Only six experiment-versus-HF layer boundaries were
captured;intermediate25..48 still need targeted attribution. The reference's
same-context50-layer repeat is complete,but that does not locate the experimental
versus reference divergence in every layer.

These comparisons include upstream HF-versus-experimental visual differences.
They are an independent **whole-chain attribution curve**,not isolated language
port error under identical teacher-forced experimental inputs. Historical
shared-visual FP32-linear teacher numbers(16.56%/7.22% under other configurations)
are different experiments and cannot explain away or substitute for this result.
No benign-pathology,downstream-video-equivalence or invented tolerance claim is
made. A model outlier coincidence alone does not rule out a bug in layers25..48.

## Structural and input checks

[Complete independent producer](<hf_multimodal74.py>) loads **903 original BF16
unsharded parameters,51,506,202,080 bytes**,and executes466 unchanged`nn.Linear`
modules with observed BF16 inputs. There is no Linear/norm forward replacement,
no identity embedding or identity finalnorm,no FP16 model conversion. Meta-model
initialization restores only nonpersistent rotary buffers from original HF CPU
constructors;checkpoint parameters load directly with`strict=True,assign=True`.

The original checkpoint has64 language layers;H3 selects retained0..49. This
intentional boundary differs from the current TP8 encoder's752 packed/sharded
parameters and7,480,909,280 bytes **per rank**. Representation counts are not
compared as though they were the same unsharded model.

[Fresh reference receipt](<hf_reference01.json>) validates actual before/after
input/model bytes and freshly measures:

- Real vocabulary embedding:all raw bits equal to the experimental encoder.
- Integer M-RoPE positions and visual mask:all raw bits equal.
- HF versus experimental merged vision: **5.6605177% relative RMS**.
- Three DeepStack inputs: **1.1360649%/2.2107090%/3.7688787%**.
- The27 text input rows match bits;the2016 visual input discrepancy equals the
  complete merged feature discrepancy. HF's own masked-scatter/position helper
  executes,not a copied experimental feature/position computation.

[Current experimental capture](<capture_summary.json>) separately verifies8
excluded complete encoder outputs and all intermediate identities against
Round73. Only rank0 writes six actual full BF16 block fixtures0/1/2/9/24/49,
**125,531,382 fixture bytes**. They are freshly captured current states,not
materially older Round54 layer fixtures. [Capture driver](<capture_text74.py>)
retains original input/model bytes and restores all wrappers/shadows at exit.
Its excluded I/O/compile forward is not a warmed latency sample.

The final HF norm is not mistaken for H3's selected boundary:it still executes,
but [post-completion proof](<reference_stability01.json>) shows its pre-input
matches raw layer49 identity. Pre-versus-postnorm differs greatly and is reported
separately,never incorporated into the H3-versus-HF result.

## Independent reference repeat and publication discipline

[Language repeat](<hf_language_repeat74.py>) uses **newly generated independent
HF** multimodal input tensors,original BF16 weights,50 unchanged HF layers,and
the same all-ones mask. This repeats the language calculation,not another fresh
vision computation. [Vision comparison](<vision_repeat74.py>) explicitly compares
two independently executed full HF vision outputs;it does not claim that the
comparison script re-executed a tower.

The repeat producer's optional early file-existence check is not the acceptance
authority. [Post-completion verifier](<verify_reference_repeat74.py>) requires
both completed/no-failure receipts before reading full tensor files. It checks
independent HF input file SHA,all50 layer identities,full pre/postnorm tensors
and layer49/pre-norm identity. Existing producer field schemas are mapped
explicitly;executed sources or failures are not rewritten.

[Full-reference aggregator](<summarize_reference74.py>) independently checks the
50 authoritative producer identities against the published stability list and
cross-validates final full/text/visual metrics from two producers(1e-12 tolerance
only for double aggregate reduction order). No tensor-element tolerance converts
a numerical mismatch into identity. Shared experimental features,FP32-adapted
teachers,partial runs,wrong masks,dtypes,positions or finalnorm boundaries fail
closed.

[Partial snapshots](<progress/hf_snapshot01.json>) are retained as partial controls,
not overwritten as completed results. Completion is evidenced by the actual
[full producer](<hf_reference01.json>),[repeat](<language_repeat01.json>),
[modal](<modal_comparison01.json>) and[stability](<reference_stability01.json>)
receipts,all completed with no failure.

## Cost,provenance and remaining gates

Full original-BF16 CPU reference computation took **5,000.89s**,including vision
about514.53s;language-only repeat took about4,461.69s. These are CPU reference
compute costs under shared-host execution,**not 910A kernel/service performance
or official warmed benchmarks**. They include diagnostic hooks and cannot be
compared to resident encoder5.384s as a speedup.

Actual pixels were already production-transport FP16. This fixture does not
cover upstream processor FP32→FP16 rounding,video tokens,loader/offload residency
or independent end-to-end generation quality. The complete independent hidden
comparison is necessary evidence,not an official-lossless certification.

[Installed-code provenance](<provenance01.json>) pins HF modeling/configuration/
processor/mask/SDPA,Torch linear/function sources,original config/index and
actual input/reference hashes. [Final verification](<final_verification.json>)
confirms23 unchanged production sources,27 executed diagnostic/dependency sources,
10 installed/configuration sources and17 real reference/selected-layer fixtures.
All match. The streaming hashes run on the remote CPU;local source/manifest
checks are distinct from rehashing absent remote tensor blobs.

[Durable final state](<STATE.json>) and[cleanup](<cleanup_receipt.json>) record
completed jobs,idle NPUs,zero accelerator processes and the owned lease released.
Production integration is unchanged. The next quality step is a same-input
25..49 text-layer attribution/replay or finer selected-boundary capture,not
bypassing the measured discrepancy to deploy further fast math as lossless.

20 new complete-reference/source/progress-gate tests pass alongside175 prior
regressions,**195 total** in[regression receipt](<test_receipt.json>). Tests pin
original BF16 weights/inputs,reference completeness,all50 repeat identities,
pre-finalnorm/PRE-DeepStack boundaries,full/text/visual metric cross-checks,
partial/shared-feature/FP32-adapter rejection,retained numerical quality failure
and final source/fixture fingerprints. None substitute for the actual CPU/NPU
numerical receipts.

```bash
python3 -m unittest discover -s artifacts/diagnostics/round74 -p 'test_*.py' -v
python3 artifacts/diagnostics/round74/summarize_capture74.py
python3 artifacts/diagnostics/round74/summarize_reference74.py
```

The overall objective remains active for independent quality repair,full service
encoder≤2s,strict DiT<15s,whole A/V identity and task/precision/warm-protocol matched
49-step comparison. The measured independent-reference discrepancy is retained,
not relabelled as a passing quality gate.
