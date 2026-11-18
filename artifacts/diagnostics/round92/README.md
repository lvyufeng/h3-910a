# Round92 — original CPU Half block25 separates input drift from NPU arithmetic

[Round91](<../round91/README.md>) made vision more accurate but worsened the
complete selected50 output from7.42%to17.04%. This round does **same-input**
isolation instead of adding another unverified full-graph patch.

## Original CPU reference gate, not a teacher shortcut

[Original HF CPU Half oracle](<cpu/original_cpu_block25_92.py>) rebuilds only
layer25 from11originaldensecheckpointtensors,convertedtoHalfwiththeirR78SHAs
verified. OriginalHFsource/config/SDPBackend.MATH/16CPUthreads/M-RoPEremainfixed.
CPUraw24→CPUblock25must**twice**reproduceR78raw25SHA`a83f92f8...`beforeany
actualNPUinputisevaluated. [Actual gate and controls](<results/cpu02/CPU_block25_reference92.json>)
pass. OriginalSDPA'selided`None`causalmaskispreserved,notreplacedbynewmath.
NoFloatlinearweights,BFgoldoutputcasts,DeepStackafter24,orfakefinalnorm.

[Actual prefix captures](<prefix_summary92.json>) are all8rank,3cases,2repeats,
stoppedafterraw24before25;eachraw24hashreproducesR91.Thecapturedstatebundle
is62.8MBandpinnedattheremotepathin [executed provenance](<executed_sources92.json>).

## What block25 actually does

[Verified summary](<summary92.json>) comparesdifferentinputswiththe**same original
CPU Half**block,thencomparesNPUvsCPUonthe**same frozen input**.

| Prefix input | Input RMS vs CPUraw24 | CPUblock25 output RMS | CPU-only amplification |
|---|---:|---:|---:|
| Native | 0.509612% | 0.545212% | 1.069858× |
| Correctedvision/native text | 0.253075% | 0.269741% | 1.065857× |
| Correctedvision/precise text | 0.252957% | 0.269577% | 1.065703× |

**No17%explosion occurs inblock25.** Theseresultsdo notsayalllaterlayersare
well-conditioned;theyonlyruleoutblamingthisspecificblockforR91'slargefinalgap.

Foractualcorrectedvision/native-text raw24,theNPUblock's**same-input arithmetic
error**is:

| NPUblock25 mode | Relative RMS vs original CPUHalf block25(same input) |
|---|---:|
| Native attention/native rows | 0.007772% |
| Precise attention/native rows | 0.007320% |
| Native attention/FloatGM row partials | 0.006660% |
| Precise attention/FloatGM row partials | 0.005243% |

FloatGMrowpartialsstillconsume**Halfinputsandweights**,carryFloatsumthrough
HCCL,andcastHalffinally;noFloatweightshadoworFloatteacher. Thejointmodehelps
thisleaf~32.5%,butallfourfullblock25vsoriginalbaselineerrorsremain~0.26975%.
Itdoesnotexplainawayoracceptthe17.04%completeencoderresultandhasnospeedclaim.
TheCPUraw24/native/otheractualprefixinputsshowthesametinyarithmeticscale.

[Same-input NPU driver](<same_input_block25_92.py>) executes4inputs×4modes×2
repeats,onall8ranks.Alloutputsagreeacrossranks,repeatsarebit-identical,and
native recovery/parameters/inputSHAsremainfixed.TheoriginalCPU-inputnative,
precise,andjointoutputsreproduceR86hashes. Theisolatednative/precise25outputs
onactualprefixinputsalsoexactlyreproduceR91layer25hashes.

## Preserved failed attempt

Thefirst [CPU CLI failure](<results/cpu01/CPU_block25_failure92.json>) reachedthe
reference-gate stage,butoptional`tensor_info=None`shadowedthefunctionwhen
constructingthemetricreceipt. ItpublishednoCPUreference/output. Theoriginal
executedsource remainsimmutable (`838027...`). A [new-root wrapper](<run_cpu_reference_fixed92.py>)
passesexplicitreviewedR84metriccallables;originalmathandSHAacceptancegateare
unchanged.No gate was weakened to obtain a pass.

## Verification and next work

30source/actualevidencetests pass.[Final verification](<final_verification92.json>)
confirms23production+13installedAPIs,originalHF/CPUgold/launcherunchanged,and
zeroR92workers.[Actual payload/source hashes](<executed_sources92.json>) pin
prefix668ad7...,NPUblock66f794...,CPUoracle838027...,andnewCPU/NPUoutputs.
No production change or newfullencoderspeedmeasurementthisround.

[Next](<NEXT.md>):originalCPUHalf25..49suffixmustreproduceall25R78boundaryhashes,
thenpropagatethesamecorrectedraw24twiceandlocatethefirstlargeerrorjump.That
controlwilldistinguishlaterinput-onlysensitivityfromaccumulatedNPUarithmetic;
itdoesnotpresupposetheanswer. Freshsuffixforkonly,nooldagentreused.
Theowned217leaseisheldfortheCPUlargecontrolandwillbereleasedafteritsworkers
end;R92NPUcomputehasended. Goalrevision16remainsactive:wholeFP16agreement,
text<2s,liveencoder/DiTco-residencyandofficialtask-matchedcomparisonunfinished.
