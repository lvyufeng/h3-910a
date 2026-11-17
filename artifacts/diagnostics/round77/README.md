# Round77 — choose a CPU FP16 reference before further optimization

## User-selected acceptance

The user finds the actual Round76 videos visually usable. Strict prompt-adherence
observations are not the same thing as optimization-caused degradation. The user
then explicitly requested **CPU FP16 precision alignment**, not forced agreement
with a BF16 graph unsupported by910A, and asked to search GitHub/Hugging Face for
a good existing baseline **before** making our own conversion.

The [new acceptance contract](<acceptance_fp16_contract77.json>) adopts that
instruction. The old40.95% original-BF16 hidden comparison remains historical
diagnostic evidence, **not a blocking FP16 acceptance gate**. No old receipt is
rewritten and no precision pass is claimed before a new FP16 reference executes.

## Search result: FP16 implementations exist, but a usable CPU golden was not found

The [bounded public searches](<search_receipts77.json>) inspected100 HF model
entries, the targeted FP16-name query and selected official/community projects.
The GitHub repository query returned eight candidates. Earlier unauthenticated
calls included a repository-rate-limit403 and code-search401; later bounded
repository/source queries succeeded. This is **not a full-site nonexistence
claim**.

### Useful GitHub FP16 prior art

| Pinned candidate | What is actually provided | Why it is not our numerical gold |
|---|---|---|
| [Amduraznak FP16 fix](https://github.com/Amduraznak/minimax-h3-fp16-fix/tree/b09897c5c3bf4af7262bb43d04a71a94563c1590) | Dense FP16 DiT branches with FP32 residual/condition/SwiGLU islands; author reports V10032GB,1120×768/124frames,about30s/step | ComfyUI-specific GPU patch; author report not reproduced here; no saved CPU FP16 tensors or full numerical receipts |
| [Star7 FP16 loader](https://github.com/star7code/minimax-h3-fp16-exact-star7/tree/27c44a7c39fc8de2d0e8eac6c5ad8032962a30d2) | Maintained loader/instance protection derived from Amduraznak; dense and quantization-aware dispatch | Derived,not an independent gold; SM80+ switches toBF16; README explicitly disclaims cross-dtype bit equality |
| [Icbears V100 patch](https://github.com/Icbears/minimax-h3-v100-patch/tree/d869f4fc4826ed4b1b1b32c862a2193da0cf8daa) | FP16 main branches/storage with documented FP32 islands and structural/argument/dtype-flow tests | Test fixtures are upstream source code,not numerical tensors; no CPU FP16 reference receipt |
| [FP16Safe](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-FP16Safe/tree/73456ad94d92d0aec9be2e392888933ce246e4a8) | Overflow protection and an in-process verification script | Published examples include quantized/pruned weights,Qwen quantization and optional few-stepLoRA; no saved independent CPU golden |

[Pinned source receipts](<port_source_receipts77.json>) and [candidate selection](<baseline_selection77.json>)
record actual fetched sources. These repositories were **not installed or
executed**. Their existence means it would be wrong to say “nobody runs H3 in
FP16”; what was not found is the complete, independently reproducible **CPU FP16
accuracy baseline** needed for our acceptance.

The community sources consistently warn that naive all-FP16 H3 may overflow in
condition projection,residual streams and certain output projections. We will
use this as design evidence,not adopt unsupported claims that power-of-two
prescaling automatically proves bit equality. Bias,subnormal range,arithmetic
ordering and RMSNorm epsilon must be checked.

### Hugging Face weight evidence

The [bounded safetensors-header reads](<header_receipts77.json>) downloaded only
headers,not tensor payloads:

| File checked | Actual header dtype counts |
|---|---|
| Official FL2VA DiT first shard |45 BF16 +9 F32 tensors |
| Official Qwen3-VL encoder first shard |43 BF16 tensors |
| Comfy-Org unpruned FL2VA DiT single file |522 BF16 +13 F32 tensors |
| Comfy-Org50-layer text encoder single file |902 BF16 tensors |
| Comfy-Org video VAE FP16 file |562 F16 tensors |

Sources are pinned [official H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA)
and [Comfy-Org repack](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/4b53a35191437de05370da548e59aeabe2a5a09a).
The inspected DeepBeepMeep/Kijai/Unsloth catalogs did not supply a clean full
FP16 DiT+encoder CPU golden. FP16 VAE files alone are not a full-model baseline;
FP8/INT8/GGUF/Turbo/few-step variants cannot silently replace the official base.
The official released base itself is CFG-distilled—this is normal and allowed;
we exclude **additional** few-step students/LoRAs,not the released H3 base.

Matching tensor counts/size is not proof that a repack has identical values. For
our reference we therefore prefer the official source rather than treating a
third-party file's name or size as a numerical guarantee.

## Selected reference

Following the user's fallback instruction,we select:

**Official `MiniMaxAI/MiniMax-H3@0919fb684102ef115126612f3fbb84d1027e7317` →
reproducible FP16 weight conversion → independently recomputed CPU FP16
reference graph.**

The external FP16 ports are useful prior art,not substituted gold tensors.
The reference has **not yet been generated**; this search stage did not launch
CPU/NPU model inference,download large model payloads or change production.

Next execution steps:

1. Verify native FP16 tensor operator coverage in the actual CPU PyTorch version
   for Linear/SDPA/LayerNorm/Conv; do not infer support from old errors or silently
   relabel an all-FP32/BF16 simulation as nativeFP16.
2. Convert original dense BF16 parameters toFP16 deterministically,keeping names,
   shapes,shard manifests and the original F32 parameter/buffer islands. Record
   source and converted hashes; no quantization,LoRA or active-layer pruning.
3. Run the independent HF/reference encoder on the **same real IDs,FP16 processor
   pixels,grids,M-RoPE and DeepStack order**, selecting50 decoder layers before
   the final norm. Do not obtain gold by merely casting the oldBF16 outputs.
4. Declare the reference's FP32 accumulation/norm/rotary/time/residual islands;
   compare against910A with the same policy. Do not reintroduce forcedBF16
   agreement as an acceptance requirement.
5. Save complete/per-layer tensors,finite checks and repeat stability. Compare
   full tensors and normal FP16 rounding discrepancies before performance A/B.
   A video that looks similar is not the numerical oracle.

CPU reference wall time is not910A performance. Kernel A/B,warmed complete
encoder and the official A2/A3 benchmark's task/thermal/full-stage qualifications
remain separate.

## Two audit corrections

1. [Pinned official FL2VA index](<search_sources/hf_official_base_index.txt>) has
   **no4-step `base_schedule`**. The earlier localRound53 metadata containing one
   must not be ascribed to the official base. Native49 evaluation count itself
   is supported; extraFastH3/FlashGen schedules are a different route.
2. [Installed runtime dtype proof](<current_runtime_dtype_proof77.json>) checks
   the actual sourceSHA`54e45b10…` already verified remotely inRound76. Its symbol
   `_BF16_DTYPE` is explicitly aliased to`torch.float16` by the default910A patch.
   Upstream `main` uses a different constant; an online symbol name is **not**
   proof that our installed DiT residuals areBF16. This source check is not a new
   runtime tensor trace,but prevents the wrong upstream code being used as one.

## Deferred diagnostic and resource cleanup

[BF16 isolation driver](<isolate_vision_text77.py>) and [fixture pins](<fixture_hashes77.json>)
were prepared before the user changed reference policy. They were **never
executed**,are not the selected acceptance baseline,and do not supply new
performance/precision evidence. The ownedts217 lease was released as soon as the
user asked to search first. All search sources are treated as untrusted data;
no fetched code,plugin or install script was run.

[State](<STATE.json>) records the selected baseline and next work. The whole
optimization objective remains active; this round determines the baseline but
does not pretend the CPUFP16 oracle or full precision acceptance already exists.
