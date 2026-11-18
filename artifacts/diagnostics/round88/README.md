# Round88 — real head72 vision attention: better leaf, slow full path

This is a scoped FP16 numerical diagnostic on owned original910A-generation ts217,
not a complete lossless/under2s encoder acceptance. Original checkpoint Half
weights and [independent CPU FP16 gold](<../round78/README.md>) remain unchanged.
No BF16 teacher cast, Floatweightshadow or new acceptance tolerance was introduced.

## Actual results

[Actual summary](<summary88.json>) and [raw receipt](<results/vision01/vision88.json>)
verify native and bias-only outputs reproduce R87 exactly,27rawlayer boundaries
permodearefinite,all3warmrepeatsarebitidentical,andnativeoutputrecovers afterward.

| Full27vision/3DeepStack mode | Merged relative RMS vs original CPU Half | Resident vision median |
|---|---:|---:|
| Native | 1.4568% | 1.479169s |
| Onecast bias only | 1.2887% | 1.490550s |
| Accurate vision attention only | 1.4439% | 3.030824s |
| Accurate attention + onecast bias | 1.1934% | 3.046727s |

The joint path gives a modest numerical gain but costs **more than2×** the native
vision time. It is **not deployed as default**, not accepted as complete precision,
and not claimed to advance the text<2s speed target. This negative performance
result is retained alongside the successful leaves; a more accurate leaf is not
by itself a correct faster fullgraph.

## Real same-QKV CPU Half MATH leaves

Captured two original image calls `[1,16,4032,72]` after the first vision block's
original projections/rotary. Repeating the native full vision reproduced their
QKV andoutputbits. CPUHalfMATHSDPA was recomputed twice on those actualHalfQKV,
withoutcausalmasking;itsrepeatsmatched. OriginalQKV SHA stayed unchanged.

| Real image leaf | Native vs CPU Half relative RMS | Custom vs CPU Half relative RMS |
|---|---:|---:|
| First image | 9.251e-5 | 1.437e-5 |
| Second image | 9.230e-5 | 1.582e-5 |

With **exact CPU Float probabilities** injected solely for the same-leaf PV
isolation, nativeFloatP@Verrorwas2.240e-5/2.202e-5. This is not a full-model
Floatweightteacher and does not replace the independent CPUHalf model reference.

[Frozen real QKV provenance](<executed_sources88.json>) pins the remote92,901,605B
payload `/tmp/h3_round88_vision01/FrozenVisionQKV88.pt`,SHA
`3fb5d524373c96de48a4be583cee87f171f5e20fdaa8ac57fd4a31cb709ce552`.
It contains both actual image QKV/nativeoutput and recomputed CPUHalfMATHoutput,
not a BFgold outputcast. The fullpayloadstaysremote;rawresultsandSHAsareversioned.

## Kernel and mathematical contract

The [fresh-fork engine](<attention/accurate_vision_attention88.py>) reuses unchanged
R85batched Half×Half→FloatGM kernels and R83runtime. Model headwidth72 is zero-padded
to128;key4032iszero-paddedto4096;scaleis**72**^(-.5). Paddedkeysarecropped
**before**unmaskedFloatsoftmax. P is decomposed intoHalfhighplus1024scaledHalfresidual.
BothPVresultsstayinFloat throughsummation beforeoneHalfcast,andoutputcrops72.
Originalprefixesandpaddingzeroscheckedoutsidewarm. ActivatedK/Varenotcachedacross
images/layers. Biasremainsaseparateswitch;noglobaltorchfunctionalpatch.

Actualwholeprobeused**436enabledcalls/10464kernel launches** and3488Float-before-Half
querychunks,withzero rejected calls. Persistentpaddedbuffersare100,401,152B,
plannedperchunkpeak341,835,776B;thesearenotfreecostsfortheeventualco-residentservice.
32enginecontracts+4rootcontracts+4actualevidencetests=40passed.

[Finalverification](<final_verification88.json>) confirms23production+13installedAPI,
CPUreference/vision/selected50payloads andlauncherSHAunchanged;allR87/R88workersended.
Ownedts217leasewasreleasedafterverificationandreadonlyprovenancecollection;
otheragents'CUDA/CIleaseswerenottouched.

[Nextprotocol](<NEXT.md>): test real patchConv3d,HalfLayerNormandGELU leaves and
CPUinitialvisioninputresetbeforeattemptingmorefullgraphchanges. ExistingR87best
native-rowaccuratetextpathstill2.898743s,notunder2s;wholeCPUFP16agreementand
liveencoder/DiT+officialsame-taskcomparisonremainactive.
