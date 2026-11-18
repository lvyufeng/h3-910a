# Next — precision sensitivity beyond layer24,not a SiLU quick fix

## Current authoritative R86 state

CPUFP16 rawlayer24 is the unchanged input of layer25; noDeepStack (only0/1/2)
and nofinalnorm is added. Original layers25..49 only were loaded on TP8 and all
outputs/inputs/parameters/recovery were verified on all8 ranks.

[Rebased results](<summary86.json>): nativeattention+nativerow0.5213%,accurate
attention+nativerow0.7214%,accurateattention+FP32row0.5450%. Compare the R85 full
CPU-condition2.4757%/1.8193%/2.7148%. Resetting the input sharply reduces final
error,but the remaining0.52–0.72% is not a pass or an end-to-end optimization.

The actualblock25 Half leaf controls disprove the anticipated large SiLU issue:
- inputnorm6.425e-6 relativeRMS versus originalCPUHalf two-roundnorm;
- local gate/up1.306e-5/1.345e-5;
- nativeHalfSiLU4.423e-7; **FloatSiLU thenHalf returns the same output bits**;
- SiLU*up3.132e-7 on same actualvalues;
- localdown1.228e-5 (noHCCLinsideleaf).
No FP32-SiLU patch was made because this real control gave no improvement.
R79 norm and Half textRoPE order were checked against installed HFsource,
including Float frequencies thenHalfcos/sin and Half multiply/add.

## First next useful control

1. Rebase from the unchangedCPUrawlayer9 (into10) as another bracket,or choose
   a single late layer25 on exactCPUraw24 input; keep originalparameters,
   positions,mask and dtype. Keep native row as the default control. Record
   rawmagnitudes/hashes/repeats and where native-versus-precise paths separate.
2. If full tensors for CPUlayer25 are needed,compute **one original HF CPU Half
   decoder block25** from the already verifiedraw24 and originalHalfconverted
   weights,not a wholeFloatteacher or a replaced gold. Require that its raw25
   hash equals the existing referenceJSON's language_layers[25].output.sha256
   before using its payload. Match the R78 original HF MATH SDPA/config/rope
   buffers/thread settings; do not change/rewrite reference.json or old outputs.
3. Compare actual block25 gates/QK/attention and row outputs on the same exact
   inputs,not only operator randomness. R86localLinear~1e-5 and tiny pointwise
   differences can amplify through FP16 late stages; a more precise leaf is not
   monotonically a better final50 result (proved R84/R85).
4. Optional CPUHalf implementation/thread variation on the same subgraph can
   quantify CPUreference numerical sensitivity,if needed. It is a separate
   experiment with preservedreference; it is not permission to waive a gap or
   pick an after-data accuracy tolerance.

The R85 joint path was4.80e-4 at24but2.71%at49; R86 exactinputreset led0.545%.
Don't assert a unique late operatorbug merely from this amplifiederror or remove
10high-range visualtokens from the acceptance metric.

## Performance and vision work remain

Bestfull CPU-condition R85control accurateattention+nativerow1.8193% has not yet
been separately full-warm timed with nativevision. CombinedFP32rowpath3.064s
is not the default; it costs extra2.16GBHalfrowweightcache/rank and regressed
fullaccuracy. A newfull test must include truevision/packing/HCCL/D2H,not CPU
precomputed conditioning in its timing.

R85textPV15×leaferror improvement is verified but originalvisionPVandbiased
Halflinearsremain. Actualvisionhidden1152/head72/MLP4304/merge4608→5120;
per-image4032keys must be explicitly padded/cropped for accuratebatchattention.
PortableFP32UBbias+oneHalfcast was verified R83,but needs realvisionweights and
zero-only4304→4352padding controls before full27vision/3DeepStack comparison.
Do not assume the texthead128/seq2043constants apply to vision.

After selecting a graph path with actual accuracy gain,fuse probabilitysplit/
packing and twoPVepilogues; don't make an FP32weightshadow or fakeFloatGMinput
asHalf to bypass the runtimeABIguard. OriginalCPUHalfroundingcontract mustremain.

## Persistence / ownership

R86source/results are immutable,allrequiredfresh-forkresult/workerjobscollected.
23production+13API/CPUgold/runtimeSHAunchanged,workersabsent,ownedts217released.
OtherCUDA/CI leases not ours. Allnewsubagents **freshforks**,never continue an
inactiveoldone. Liveclaimowned beforecompute;noCI/dormantopt-in. Goal16 active/
armed; wholeCPUFP16agreement,text<2s,liveencoder/DiTserviceco-residencyandofficial
same-taskA2/A3benchmark comparisonstillunfinished. PriorDiT/VAEresultscope
qualificationsremain.
