# Round91 — actual full TP8 result: cache is bit-exact, vision combination rejected

This round measures the **real** private FP16 `encode_ids` path: actual pixels,
27visionblocks,50textlayers,layout/attentionpacking,HCCLandD2H.NoCPUprecomputed
imageconditionisusedinthefullwarm.[OriginalCPUHalfreference](<../round78/README.md>)
andparametersremainunchanged.NoFloatrowadapter/Floatlinearweightshadow.

## Full eight-rank measurements

[Actual summary](<summary91.json>) verifiesfivepairedwarmrunsforeverymode:
barrierbeforetheclock,thenpercallmaxof8ranks,thenmedian.Hooks/hash/CPUcomparisons
areoutsidetheclock;allwarmoutputsreproducethecase'sstock/equivalentoutputbits.

| Actual complete encoder path | Raw selected50 RMS vs CPU Half | Paired8rank max median |
|---|---:|---:|
| Native | 7.4201% | 2.277611s |
| Native + geometrycache | 7.4201% | 2.207390s |
| Correctedvision + native text | **17.0374%** | **2.064467s** |
| Correctedvision + native text + geometrycache | 17.0374% | 2.085461s |
| Correctedvision + precise text + geometrycache | 14.8014% | 2.736972s |

**No mode is under2s**,andthecorrectedvisioncombinationmustnotbedeployedasdefault:
itswholeencoderprecisionisworse,despitereducingvisionmergedRMS1.4568%→0.7244%.
Theslowerprecisetextattentionhelps17.04%→14.80%butdoesnotmakethefullgraphpass.
Noaccuracytolerancewaswaivedorselectedafterseeingthesevalues.

## What remains a valid lossless optimization

The [geometry cache](<cache/geometry_cache91.py>) returnsoriginaldeterministic
position/rotarytensorscomputedatprime,onlyforthecurrentgrid/unchangedweights.
Itcachesno pixels/patches/visionhidden/DeepStack/textconditions.Atall8ranks:
- Nativecacheon/offoutputsand**all77rawboundaries**matchbits.
- Correctedvisioncacheon/offoutputsandall77boundariesalsomatchbits.
- Originalparameters/grid/fixtureandcachedtensorSHAsremainunchanged.
- Weight/config/invfreq/grid-valueguardsfailclosed. Rootreviewaddsforcedvalue
  validationforunversionedinferencegrids;pointer/objectidentityneverstands
  inforversionedcontents. Newstockforwardgridallocationsareactuallytested.

Thepurecacheimprovesnativecompleteencode2.277611s→2.207390s(~3.1%)butcannot
byitselfreach2s.Correctedvision'scachepairedmedianisslightlyslowerinthissample;
noextrapolatedspeedbenefitisclaimedthere.Extracache**19,740,672B/rank**.

## Why smaller vision error is not whole-text acceptance

[Actual raw-boundary drift](<layer_drift91.json>) shows the interaction:

| Original raw text layer | Native RMS | Correctedvision/native text RMS |
|---|---:|---:|
| 0 | 1.3154% | 0.6536% |
| 9 | 0.3493% | 0.1753% |
| 24 | 0.5096% | 0.2531% |
| 49 | 7.4201% | **17.0374%** |

Early/midlayersimproveyetlateerrorsamplifydifferently.Thisdoesnotproveoneunique
remainingoperatorbug. Thewhole-referencehigh-range/visualsubsetsarepreservedin
rawreceipts;thetext-nonvisualsubsetdoesimprovefrom7.39e-4to3.22e-4,butthatisnot
usedtohidebadvisual/high-rangeoutputsinwholeacceptance.

WithoriginalCPUHalfinitialtext/DeepStackconditions,theunchangedcontrolsreproduce
R82/R85: native2.4757% / precise1.8193%.Thoseare**accuracyisolationsonly**,not
completeencoderwarmtimings. AllsixoriginalCPUmid-layerpayloadSHAsverifyagainst
R78;rawlayer49pre-finalnormistheactualH3boundary,notoptionalfinalnorm.

## Execution and safety

[Serialprebuild](<results/full01/prebuild91.json>) verifies4text+7visionbias+1erf
objects before8rankmodelpriming.NofakeA2SoCorBF16support,unchangedR79/R81/R85/
R87/R89/R90sources/binaries.[Executedprovenance](<executed_sources91.json>) pins
actualsources.[All8rankreceipts](<results/full01/ranks/tp8_rank0.json>) includerank0
index;theother7areindividuallySHA-pinnedinthefullsummary.

40contracts/actualevidencetests passed.[Finalverification](<final_verification91.json>)
confirms23production+13API,CPUgold/vision/selected50/HFandlauncherunchanged;all
workersended.Owned217andfreshunusedCUDA133/134grantsreleased;others'leasesuntouched.

[Nextprotocol](<NEXT.md>):same-inputoriginalCPUHalfblock25isolationandinput-only
error-amplificationcontrolbeforemorefullvision/textmixtures;separatebit-exact
metadata-speedwork.Goalrevision16active:wholeFP16agreement,text<2s,liveencoder/DiT
co-residencyandofficialtask-matchedA2/A3pipelinecomparisonremainunfinished.
