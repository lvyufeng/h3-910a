# R134 — user-authorized local FP32-output/bias candidate test

[Approval](approval134.json) authorizes **only a local original→candidate→restored
comparison**, not full50 regression,49-step generation,baseline adoption,tolerance
orcopy optimization. [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) stays
InProgress; [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) remainsBlocked.

## Recorded local outcome

[Actual results](RESULTS.md) and [verified numerical summary](summary134.json) show
all8candidate rankgroups repeat exactly for128local/32modulecalls. Nativeandrestored
paths stilldrift byoneHalfword. Butcandidatechangesup5447/16128localHalfwords or
6484/32256postgatherHalfwords, maxabs9.765625e-4 andpostgatherrelRMS0.01595574%.
Thisisnot a lossless replacement; baseline/thresholdnotadopted,full50 nottested.

[Cleanup](cleanup134.json) records allprivateprocessesexited,parameter/sourcepins
unchanged,False/warnFalse andmethoddescriptors restored;ownedlease released.
Anyfull50candidate diagnostic needsnewexplicituserauthorization; copyremainsoff.

[Candidate](fp32_bias_candidate134.py) keeps sameHalf x/weight andoriginalHalfbias.
It calls`torch.mm(x,weight.t(),out_dtype=float32)`,adds`bias.float()` inFP32,then
narrows once toHalf. **This changes originalrounding/bias behavior** andisnot
assumedlossless or stable. Unsupportedruntimeerrors stopthecohort; nofallback
Halfmm-upcast ispresented asFP32 GEMMoutput.

[Probe](candidate_probe134.py) keeps originaltime/finalAdaLN components,actualTP2+USP4,
originalreferenceweights/fixtureSHA and R132actualt_embSHA gate. SameprivateTrue
flag isusedforallthreecase comparisons,thenrestoredFalse. Eachcasehas32smallModule
calls(withTPgather) and128directlocalcalls(noTPgather). Neither isafull50or155pass.

WholeHalfword comparisons use separatecase-first/original-first references. Original
Module-first anddirectlocal-first aredistinct anchors. Everymoduleinputmatchesthe
originalCPUfirst; thefixedNPUcloneisword-matchedbeforeuse. Originalparametersbefore/
after mapsaresaved,descriptor andflagsrestore areseparatefromnumericalbitagreement.
Successfulcandidatecallcounter160/rank is aPythonpathcount,notCANNkernelattribution.

[Mock unit](test_mock134.py) checksonly callorder/dtype/interface andparameteridentity,
notnumericalequivalence orNPUrepeatability. [Portabletests](test_contracts134.py)
checksource/contracts. [Launcher](launch_candidate134.py) hasnewexactroot/capability,
cohortfatalmonitor,total20-minbudget andPIDfd-onlyprivatecleanup. NoSDK/driver/device
reset/sharedprocess/CI/dormant changes. Weights/inputs/activations stayonthehost,
onlysource/JSON/textreturned. Actualresultsarekeptseparatelyafterexecution.
