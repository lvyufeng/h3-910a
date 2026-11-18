# Next — late-layer precision isolation,not blind FP32 operator stacking

## Verified current state

R85 captured the real original decoder0 Half Q/K/V on all8 TP ranks with exact
CPUFP16 input. CPU Half MATH attention is the unchanged same-operand leaf
reference. Native score error~2e-7; even with exact CPU probabilities injected,
native Float P@V error~1.23e-4–1.67e-4. The batched Half→FP32-GM kernel plus
P_hi/1024-scaled-P_residual reduced actual attention error2.340e-4→1.547e-5
(15.1×),with repeat bits and unchanged operands. P reconstruction error~3.5e-8.

[Verified engine](<attention_engine85.py>) prepacked K/V once per SDPA call and
matched the frozen leaf result bit-for-bit before fullTP8 loading. It is scoped
to text SDPA only; original vision Float PV and biased Half linears unchanged.

Actual full50 CPU-condition results in [summary](<summary85.json>):
- native attention/native rows:2.4757%;
- accurate attention/**native rows**:1.8193% (best current same-CPU control);
- accurate attention/FP32 rows:2.7148% (worse than original baseline);
- latter with native vision conditions:7.6094%.

All8 repeats,original native output recovery and frozen parameter/input SHA
gates passed. Combined accurate-attention+FP32-row genuine warm encode3.064183s
(not <2s). Best native-row path has not yet been separately timed for full encode.
Don't deploy the combined path or misstate leaf precision as whole-graph success.

## First next experiment: rebase late stages onto an unchanged CPU gold state

[Layer drift](<layer_drift85.json>) shows the joint path is better through layer24
(4.80e-4 vs native5.19e-4),but worse at49(2.7148% vs2.4757%). The first real
same-CPU-input block itself improves1.457e-4→7.03e-5. This localises the observed
error amplification to the late24→49 interval without proving its cause.

Use the already saved **original CPUFP16 raw layer24** payload as an exact
input to NPU layer25 and run25..49,with original parameters and positions:
1. Native row/native attention; accurate attention/native row; optionally FP32
   row separate. Recordallraw25..49hashes,magnitude/finite summaries and selected
   boundary comparisons against the same storedCPUgold. Repeat each case.
2. Since all3DeepStack injections are after0/1/2,raw layer24 is directly the
   next input; **do not add DeepStack again or apply final norm**.
3. Require bit-exact CPUlayer24→NPU transfer and frozen original input/parameter
   hashes. Compare finalraw49with the unchangedFP16_layer49/pre_final_norm gold.
   This is a precision isolation experiment,not a full-encoder timing shortcut.
4. Optionally rebase from rawCPUlayer9 to layer10 to bracket the amplification.
   Don't modify/recompute the CPUgold unless additional checkpoints are truly
   needed; its fullFP16 reference is already valid/repeated.

Also inspect actual late-layer leaf operators on fixed inputs rather than
ascribing error to a dtype label. For the late stage where divergence starts:
- CPU Half vs native Half q/gate/up linear outputs,including real shape and values;
- actual Half SiLU then Half multiplication versus CPU Half semantics;
- Float normalisation → Half normalised value → Half gamma multiplication (R79
  already follows this order; don't change it into a one-round Float-gamma norm);
- RoPE angles/outputs,Float softmax and compensated PV.

A small leaf error can amplify on sensitive large visual-token states; R85
observed10 high-range tokens dominate finalRMS. This is not permission to ignore
those tokens or choose a new tolerance after the data.

## Independent vision precision work still needed

R79 native vision merged output differed1.46%,and original vision PV/biased Half
linears are still used. R85 text SDPA fixes do not fix vision by implication.
- Actual vision: hidden1152,head_dim72,MLP4304,merger4608→5120,M8064/2016.
- Test original Half biased linears against CPUHalf `F.linear` on the same real
  frozen operands using R83 FP32UBbias+oneHalf epilogue. Padding4304→4352 must
  be zero-only with exact prefix/crop controls; no FP32 weight shadow.
- An accurate vision attention variant needs explicit head72 zero-padding to128,
  per-image key4032 zero-padding to4096,head16/image2 batching and output crop.
  Preserve Float P high/scaled-low arithmetic and CPUmath reference; do not just
  pass text head128/seq2043 constants into the vision path.
- Keep separate row/attention/vision switches,restore native baseline between
  A/B cases. Default row remains native because the FP32-row combination regressed.

## Performance follow-up after precision selection

Prototype batched leaf15.737ms includes Python padding/packing,3GEMM launches per
querychunk and twoFloatPVoutputs. Full combined encode3.064s. No speedup is claimed.
Potential fusion: a single PV kernel with Hi/Lo Half inputs,sharedHalfV,FloatL0C
for both products,FloatUB weighted add and oneHalf GM output; no Half partial.
Another candidate is splitting/packing FloatP into Hi+scaledLo in a device kernel.
Do not fake Float input as Half merely to bypass the5GMbridge dtype guard; use
an honest versioned GM-role bridge if needed and keep the old library pin intact.
Measure actual best native-row full encode first,including vision/layouts/HCCL/
D2H and memory;don't substitute CPU-precomputed conditioning for timed vision.

## Persistence and ownership

Full R85 sources/results are immutable. 23production+13API/CPUgold/runtimeSHA
verified unchanged; related workers absent; ownedts217 released aftercompute.
Other CUDA133/134 sessions' leases are not ours. Liveclaim owned before any NPU
work;noCI/dormant opt-in. Every new subagent must be a **fresh fork**,never resume
inactive old ones. Goal16 remains active/armed; none of the full objective,
CPUFP16 graph agreement,text<2s,live encoder/DiTco-residency or officialtask-matched
A2/A3 comparison is claimed complete. EarlierDiT/VAEmeasurements retain original
scope qualifications.
