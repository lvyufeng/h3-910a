# Round89 — real CPU Half entry, patch GEMM and nonlinear controls

The original [CPU FP16 reference](<../round78/README.md>) remains unchanged. This
round adds a real one-cast patch projection on original910A-generation ts217,
not a BF16 comparison, Floatweightteacher or whole-encoder precision acceptance.

## What actually improved

[Actual summary](<summary89.json>) and [fixed raw receipt](<results/vision02/vision89.json>)
compare all27visionblocks/3DeepStackoutputs with the original CPUHalf gold.

| Full vision path | Merged relative RMS | Resident vision median |
|---|---:|---:|
| Native | 1.4568% | 1.476421s |
| Patch onecast GEMM only | 1.4218% | 1.371665s |
| Patch + onecast vision linears | **0.7246%** | **1.381477s** |
| Exact CPU entry + onecast linears (diagnostic) | 0.7067% | not a speed candidate |

The joint real path nearly halves the whole-vision error and improves its actual
resident time by about6.4%. It gets close to the CPU-entry-reset control without
using a precomputed CPU entry in the real path. Yet **0.7246% remains unaccepted**
and this is not a complete text<2s benchmark. No new tolerance was selected.

The [patch adapter](<patch_bias89.py>) proves the original biasedHalfConv3d has
kernel=stride=(2,16,16),groups1 andoneoutputvoxelperpatch. Its same original Half
input/weight+bias become M8064/K1536/N1152,then the verifiedR87biasbuilder performs
Half×Half→fullKFP32L0C→Floatbias→oneHalfcast. Layoutprefixeschecked; no extra
M/K/Npaddingneeded; no Floatweightshadow. Real CPUHalfConv3d errors:

- Native: **1.63136e-4**.
- New patch GEMM: **2.45796e-6**,about66×lower.
- CPUHalfflattenLinearvsCPUHalfConv3d: **2.39909e-6** (not assumed bitidentical).
- NewGEMMvsCPUHalfflattenLinear: **7.10711e-7**.

14 actual patch kernel launches were verified. NewHalfweightlayoutcache is
3,538,944B andFloatbias4,608B,besidesR87linearlayoutcache; neither is free for an
 eventual co-resident service. [Execution provenance](<executed_sources89.json>)
pins the new5304BkernelobjectSHA and both failed/fixed sources.

## Original CPU entry is independently pinned

[CPU entry helper](<original_hf_entry89.py>) loads all351originalHalfvisionparams,
uses the originalHFsource/16CPUthreads and recomputes patch/positions/entry twice.
Then originalCPUHalfMATHblock0reproducesR78's rawblock0SHA exactly:
`1b41bf75401dfbcef77dff545c690857ee4ff7b58a8d49c6eadf13c6fbfdd5ef`.
PositionsalreadymatchCPUbits. The nativeentryerror1.76843e-4 is real,but entry
resetaloneonlychangesfullvision1.4568%→1.4165%; its interaction with onecastlinears
isimportant,not a uniqueentry-onlyrootcause. CPU reset runs contain external
referenceinputandD2Hchecks,so theirtimesare**not**realencoder speed candidates.

## Norm/GELU Float widening does not help

The [fresh-fork controls](<leaf/vision_leaf89.py>) test7realHalfinputs with original
Halfgamma/bias,eps1e-6 and originalMLPtanh/mergererf modes. NativeLayerNormand
blocktanhGELUerrorsareonly5–7e-6; FloatthenHalfdoesnotimprovethem. Allnorm/GELU
widening fullvisionvariantsreproduce nativeoutputSHA,andaddingthemto the joint
patch/linearbiaspathalsodoesnotchangeitsoutput. These redundantFloatpathsare
**not**keptasdefaults.

Merger `GELU(approximate='none')` stilldiffersfromCPUHalferfGELUby3.0977e-4,
firstDeepStackmergerby4.3318e-4. NPUFloatthenHalfhasthesamegap,whileCPUFloatthenHalf
onlydiffersfromCPUHalfby2.89e-7/1.13e-6. This motivatesanexplicitCPU-erf/NPU-erf
leafcontrolnext;itdoesnotyetprovewhichbackendimplementationisresponsible.

## Failure and verification

The [initial receipt](<results/vision01/vision89.json>) and [initial log](<results/vision01/run.log>)
are preserved: originalCPUblock0SHAalreadyverified,butrootcalledtheleafcontrols
outsideinference mode andautogradrefusedsavedinference tensors. The
[new fixed driver](<vision_probe_fixed89.py>) adds that context in a new remote
root; originalHF/reference/arithmeticwasnotrelaxed. Full11cases nowfinish,all27
boundaries finite,all3warmrepeatsmatchbits,nativeoutputandalloriginalparam/input
SHAsrecover.29source+actualtests passed.

[Final verification](<final_verification89.json>) confirms23production+13installedAPI,
originalHFsource,CPUreference/vision/selected50payloadsandlauncherunchanged;
noR89workersremain. Ownedts217leasewasreleasedafterreadonlyprovenancecollection.
The310,953,441Bfrozenentry+leafpayloadstaysremote with its SHA in theprovenance.

[Next](<NEXT.md>): explicit erfGELU control/kernel and real completeTP8encoder
measurementwiththeusefulpatch+biaspath,notFloatwideningorR88slowattention.
WholeCPUFP16agreement,text<2s,liveencoder/DiTco-residencyandmatchedofficial
A2/A3pipelinecomparisonstillremainactive.
