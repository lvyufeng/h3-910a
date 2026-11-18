# Next — upstream accuracy now takes priority over blind late-layer kernel patches

> **Post-run update:** [Round95](<../round95/README.md>) is now complete: CPU-exact
> RoPE wholegraph precision is non-monotonic, no case is under2s, allworkersended
> andowned217released. Use [latest next protocol](<../round95/NEXT.md>); the
> running-job instructions below describe the earlier handoff, not a live job.

## Verified CPU-only explanation

[CPU93 complete receipt](<results/suffix01/CPU_suffix_reference93.json>) reproduces
all25R78rawboundarySHAsfromoriginalCPUraw24,thenpropagatestheactualcorrected
NPUraw24e232...twice withoriginalCPUHalf25..49and275originalHalfparameters.
Input0.253075%→raw4201.052459%→**raw4316.648505%**→raw4917.161829%.
Thelargestjumpis43(15.8187×),totalinputamplification67.8133×. So17%doesnotneed
NPUarithmetic toexist. Itdoesnotidentifywhich43primitiveisresponsible,andit
isnotvalidtosubtractCPU17.16%fromNPU17.04% asanadditivearithmeticfactor.

R94sameCPUvariance NPUrsqrtisbit-exact;NR1/2slightlyworseonactualHalfNorm5120.
NoNRpatch.TextHalfcos CPU/NPUdiff7.14e-6/oneHalfULP;R79frequencyFloatmatmulerror
wasalreadycorrectedbydirectFloatmultiply.CPUpreciseRoPEtableinterventiononly
changesoriginalCPUconstanttrigresults,notconditionsorlearnedweights.

## Round95 actual full A/B now running

[Round95STATE](<../round95/STATE.json>) is the authority for currentjob/sources.
Freshfork`cpu-rope95-fresh`task46 sourcecompleted andRootreviewed26tests;code
`cpu_text_rope_cache95.py`4e7cabcf...disabledbydefault. Sixfullstockencode_ids
cases actualpixels/27vision/50text/HCCL/packing/D2H,5paired8rankmaxsamples after
CPU93workersended. NativeRow/noFloatweights,geometry19.7MB cachealwaysenabled
(same77nativebits),CPUtextRoPEtable1.046MB old/new fornativevision,correctedvision
nativeText,andcorrectedvisionpreciseText. NoRMSNR/R88slowvisiondefault.

Do notassumeCPU-exactcos/sin fixes43'scliff;fullprecisioncouldstaybadorworsen.
Allraw50/high-range/visualdiagnosticsandbadR91baselinepreserved. Originalnative
R82SHA,oldcorrectedR91SHA,repeat/native recovery/unchangedparameters andCPUgold
mustmatch. CPUinitialtextconditionsare accuracycontrolsONLYnot encoderwarm.

Owned217expires20:21:10Asia(`2026-11-18T12:21:10+00:00`). NPU95supervised1200s
startsaround19:40afteroldCPUended;finishbeforeexpiry. Collectevery relevantjob
and all8rank JSONs/logs. Ifnonzero readfailurebeforechanges,newroot/no uncertain
replay.Do notsilentlyreusedeadlease. Final23production+13API/oldgold/HF/library,
all95/93/94workerszero thenrelease217.Otherleasesuntouched/noCI/dormant.

## Technical priorities after CPU-exact RoPE A/B

1. **Accurate and fast vision attention.** ExistingR88genericQK/high+lowPVhelps
   frozenleaf6×buttooslowandwholevisionnotyetprecisionpass.Testitonlyin
   controlledA/Bwithcurrentpatch+bias+erf,retainlatencynegatives. Astreaming
   fusedFAneedshead16,D72/originalscale72,key4032,unmaskedvision,stableFloat
   onlineLSE,Halfprobabilityhigh+residual1024lowPVFloataccum+singleHalfoutput.
   No causal mask/noA2 ISA/noFloatweights. Dpad80(zero8)couldsaveworkvs128butis
   a newshape needingexactpacking/batch/last64key/query-tailproof. Do notcallit
   donebeforeactualR88frozenQKVSHAs/CPUHalfMATHleaf/fullvision/fullencodergates.
2. **If instrumenting CPU43**, originalCPUraw42/43stateswereNOTsavedby93,only
   theirverifiedSHAsandfinal49outputs. RecomputeCPU25..42 onbaseline and
   correctedinput(to43oneblock)innewroot,requireold93SHAs,thenoriginalCPU
   layer43hooks(attentionvsMLP/postnorm/qkv/probabilityrows)isolateswhichpath
   amplifies. ThisisnotanNPUkernel-bugpresumptionandnoliftedFinalNorm.
3. Maintainprecision-speedseparation:originalnativegeometrycache2.207s/R91
   fastestbad-corrected2.064s stillnot<2. Newunder2claimrequiresall5pairedrank
   maxima andactualrealencode.Donot hideimageconditioningorCPUgoldinwarm.

Do not resurrect old inactive agents; every delegation is a fresh fork. Do not
changeR78originalCPUFP16gold/FloatLinearweights/tolerance. Wholegoal16active:
completeFP16agreement,text<2s,liveencoder/DiTco-residencyandmatchedA2/A3pipeline
comparisonremainunfinished;R76DiT49mean14.657s/VAE13.5shavetheiroriginalscope.
