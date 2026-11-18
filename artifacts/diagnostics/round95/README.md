# Round95 — CPU-exact RoPE leaves do not guarantee whole-encoder agreement

[Actual summary](<summary95.json>) verifies6cases×8ranks,each5pairedwarmcalls;
realprivate`encode_ids`includespixels/all27vision/all50text/HCCL/layout/packing
andD2H.CPU93largecontrolendedbeforetheseclocks. Geometrycacheisalwaysenabled
(thebit-native19.7MBcache);notanimageconditioningcache.

## Complete encoder A/B

The [CPUtextRoPEcache](<cache/cpu_text_rope_cache95.py>) recomputesoriginalHFCPU
Halfcos/sinfromrealpositions **withoutreadinghiddenvalues**. 0learnedparameters,
noFloatLinearweightteacher/noBFgoldcasts. CPUtablescomputedtwice,byte-exactR94
CPUtableSHAs,andcopiedtoNPUunchanged.Extra1,046,016B/rank.Defaultdisabled.

| Fullrealpath | CPU-exacttextRoPE | Selected50RMS vs CPUHalf | Paired8rankmax median |
|---|---|---:|---:|
| Nativevision/native text | off | 7.4201% | 2.136741s |
| Nativevision/native text | on | **7.4895%** | 2.158240s |
| Correctedvision/native text | off | 17.0374% | **2.037664s** |
| Correctedvision/native text | on | 13.6159% | 2.062419s |
| Correctedvision/precise text | off | 14.8014% | 2.669211s |
| Correctedvision/precise text | on | **17.1336%** | 2.686949s |

**No case is under2s**,not even thefastestindividualsample2.006450sintheCPU-table
correctedvisioncase. Thecacheimprovesonewholecasebutworsensnativeandprecise-text
cases;itisnotageneralprecisionfixandnotadefaultspeedoptimization. Complete
FP16CPUagreementisnotacceptedandnoafter-the-facttolerancechosen.

Theoldnative/oldcorrected/oldpreciseoutputSHAsexactlyreproduceR82/R91despitenew
clockvariation.Donotattributecross-round2.207→2.137stoanewoptimization:the
same-runold/newpairedvaluesaretherelevantcomparison. Alloldandnewnumberskept.

## CPU-conditioning controls are not full encoder timings

OriginalCPUHalftext/DeepStackinputspass unchanged ontoNPUtext50foraccuracyonly:

| Textattention | OldnativeRoPE RMS | CPU-exactRoPE RMS |
|---|---:|---:|
| Native | 2.4757% | **3.0516%** |
| Precise/native rows | 1.8193% | **1.9973%** |

ThesealsoworsenwiththeCPUtable.TheR94one-ULPcosmismatchisrealandthetable
removesitbit-exactly,butthatdoesnotmakewhole-graphroundingerroramplification
monotonic.[CPU93](<../round93/README.md>) alreadydemonstratedthatoriginalCPUHalf
alonecanamplify0.253%inputerrorto17.16%,withlayer43firstjump15.8×.
No NPU arithmetic is required to reproduce that scale.

## Verified structure, not a waived precision result

- All8ranksrepeat everycaseandstock/equivalent outputbits;5warmsproducecaseSHA.
- RoPEtableSHA/version/config/source/positionvalueguardsstayunchanged;inference
  positionswithoutversionsarevaluevalidated,nottrustedbydata_ptrreuse.
- **Everyvisionraw27boundarybit**isunchangedbytextRoPEold/new.
- Originalparameter/fixture/CPUgoldSHAsremainfixedandnativegraphisrestored.
- NoFP32rowadapter/weightshadow,R94negativeNewtonRMS,orR88slowvisiondefault.

[Executed source/binaries](<executed_sources95.json>) pinsactualdriverandCPUtable
source4e7cabcf...,and12unchangedverifiedobjects. [Final verification](<final_verification95.json>)
confirms23production+13API,originalCPUgold/HF/launcherunchangedandzero95/93/94
workers.Alljobscollected;owned217releasedbefore20:21Asiaexpiry;freshunusedCUDA
133/134werealsoreleasedandothers'leasesuntouched.NoCI/dormant.

35cache-source/root/actualevidencetests pass. No separate deliverable or official
A2/A3benchmarkisclaimed. [Next](<NEXT.md>):fast CPU-faithfulvisionattention and
layer43input-sensitivitycontrols,notmoreblindcos/normpatches.Wholegoal16active:
FP16agreement,text<2s,liveencoder/DiTco-residencyandofficialmatchedpipelineopen.
