# R130 — original projection stage diagnosis

Live [Issue #15](https://github.com/lvyufeng/h3-910a/issues/15); copy-stream
[Issue #4](https://github.com/lvyufeng/h3-910a/issues/4) remainsBlocked.
No algorithm/precision policy is accepted by these diagnostics.

[Original-stage script](time_stages130.py) executes the original guard under real
TP2+USP4/world8, with original FP32time parameters. It observes four Halfoperand
GEMMs with FP32outputs and three scaledFP32additions plus actualrank0biasaddition.
Onlyoriginalresultsreturned; [snapshotobserver](projection_observer130.py) retains
independentbytes immediatelybefore/aftermutation. D2Hsynchronizesandisnonclean.
[CPUunit](observer_unit130.json) verifiesmutation snapshots andexceptionrestoration;
[portabletests](test_contracts130.py) verifysequence/alpha/originalreturn contracts.

## Actual first-stage findings

[Stage summary](stage_summary130b.json) verifies all8rank finished. Thefirstchanged
operator is **MM0** with **identicalactualHalfoperandwords** andFP32output. Fixed
same-operandreplays ofall4GEMMs showdrift inall32rank/opgroups. All28fixed-input
scaledaddition/bias groups arebit-identical. Thisisolates at leastoneoriginal
non-repeatable GEMM path; internalCANNmatmulalgorithm/split/reductionstillUnknown.

ThefirstMM0maxabsdiffis5.9604645e-8; otherfixedGEMMoutputs differ up to1.1920929e-7.
These smalloriginaldeltas are **not** permissiblecandidateerror thresholds.
Fullmodelpropagation isnotretestedhere, no155checksareclaimedforsmallcomponentonly,
andnofull49generation/copycandidatewasenabled.

## Failures preserved

The firststage diagnostic wronglyexpected3addsforallranks andfailed theevenTPprimary
ranks. InoriginalRowParallelFP32timeprojection, rank0haslocalbias,rank1none; the
observer mustinclude that originalbiasratherthanomit it. [Exact initialsources](executed_sources130a/time_stages130.py)
andfailedrankreceiptsremain; [secondpins](launch_pins130b.json) correctonlytheobserver
expectedsequence. Theoriginalformula/returns/weights neverchanged.

The secondcomponentrun succeededbutscpfailed becausetheclientdestinationdirectory
wasmissing. It wascreatedandthesamecompletedreceiptscollectedwithoutnewmodelcompute.
Noresultisoverwrittenbyaretrial. Allcomponentprocesseswereverifiedexited.

## Deterministic probe completed — not an adopted policy

[Private deterministic probe](deterministic_probe130.py) completed native false → true →
restored false on all8 ranks. [Summary](determinism_summary130.json) shows 0/40 stable
rank/operation groups in native,40/40 with theflagtrue,0/40 afterrestore.
The full localguard differs from native bymaxabs5.9604645e-8 andrelRMS4.37294e-8;
thisstable resultis **not original-bit-identical**,so themodelflag remainsfalse.

[Full findings](RESULTS.md) preserve originalstage/fixed-operation proof, allfailed
attempts, scoped timings and exactexecution sources. [Cleanup](cleanup130.json)
confirms process exit andflag/function restoration. No fullmodel49generation,
copycandidate, modelprecision pass orperformancegoal completionisclaimed.

A full-50 deterministic **diagnostic only** requires theuser'sdecision. It doesnot
authorize adoption orusingoriginalnative noise as a newcandidate tolerance.
