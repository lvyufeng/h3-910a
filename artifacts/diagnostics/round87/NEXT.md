# Next — measure vision P@V directly with real head72/key4032

## Verified R87 results

Actual qkv/mlp1/mlp2 biasedHalfLinear CPUHalf errors fell31–51× with FP32bias
beforeoneHalfcast. Complete27vision mergederror1.4568%→1.2887%,DeepStackimproved
modestly; residentvision1.4737s→1.4884s,notaspeedup. ExtraHalfcache1,186,725,888B
plusFloatbias1,247,744B. Originalparameter/input/CPUgold frozen; native recovered.

Bestnative-row accurate-textattention completeTP8encode2.898743s,notunder2s.
ExactCPUcondition1.8193% error; nativevisioncondition7.5915%. NoFP32rowcache.

Do not deploy these by themselves as completeprecision/speed fixes. Use
[summary](<summary87.json>) and [actualvisionreceipt](<results/vision01/vision87.json>).

## Current next implementation and real leaf test

Task32freshfork owns an accuratevisionattentionengine in Round88; do not resume
inactive vision87-fresh. Root keeps newRound88sources separate; allR87executed
sources remain immutable. TheR87owned217lease washelduntilR88followup,nowreleasedafterR88source/worker
verification. NewNPUworkneedsanewownedclaim;donotreclaimanotheragent'sCUDA/CIleases.

Originalvisionhelperperimage receivesHalfQ/K/V [1,16,4032,72] after original
Floatrotary+Halfcast. UnmaskedvisionSDPA,NOTcausaltextSDPA. Twoimages invokeit
separately. FirstcaptureactualfirstblockQKV fromnativevision plusoriginalCPUHalf
MATH SDPA on the same frozen values; repeatedgold and parameterhashrequired.
No Floatweightteacher, no BFgold cast, no new acceptance tolerance.

ReuseunchangedR85batch kernel546cd... to test:
- Q/K/V headwidth72 zero-pad128; keyaxis4032 zero-pad4096;
- querychunks512×7+448last,paddedM512;
- H16 batched scores Half×Half→FloatGM,**scale72**^(-.5),cropkeyaxis4032 before
  Floatsoftmax,no causal mask;
- FloatP→Halfhigh +1024scaledHalfresidual,zeroKpad4096;
- twoFP32PVresults summedinFloat beforeoneHalfoutput,cropheadwidth72;
- packedK/V refreshedpercall,not cachedacrosslayer/image inputs.

ThematchingbinaryshapesH16,M512/448,K128N4096andK4096N128are4newidentities;
originalR85batch source itselfmustnotbeedited. Prebuildseriallybeforemodel,
validateleafagainstactualCPUHalfMATH and all padding-prefix/zero controls.
Then full27vision/3DeepStack separate modes: native,accuratePV only,accuratePV+
onecastbias. Recordrealresidenttimes/memory; allrepeat/native-recovery gates.
Onlythenconsiderfull50TP8encodingwithactualnewvision—notprecomputedconditions
insideitswarmtimer. R87modestgain doesnotprovePVistheonlyremainingcause.

OverallCPUFP16agreement,text<2s,liveencoder/DiTco-residencyandofficialA2/A3
same-taskpipelinecomparisonstillactive. R76DiT49mean14.657s/VAE13.5sretain their
originalscope qualifications. Goal16active/armed. Allnewagentsfreshforks.
