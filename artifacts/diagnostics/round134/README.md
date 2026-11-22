# R134 — user-authorized local FP32-output/bias candidate test

[Approval](approval134.json) authorizes **only a local original→candidate→restored
comparison**, not full50 regression,49-step generation,baseline adoption,tolerance
orcopy optimization. [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15) stays
InProgress; [Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) remainsBlocked.

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
