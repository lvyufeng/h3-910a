# Round78 — independent CPU FP16 encoder reference in progress

The user-selected acceptance is CPU FP16 agreement,not forced BF16 agreement.
The public baseline search is complete and selected reproducible conversion
from the official H3 source. This round executes the next step;it does not
relabel an old BF16 output as an FP16 teacher.

## Verified before model execution

The [actual CPU FP16 preflight](<preflight.json>) passed all9 cases under
PyTorch2.10.0+cpu and Transformers5.14.1: Linear,causal/noncausal SDPA,
LayerNorm,original HF RMSNorm,Conv3d patch embedding,bilinear/bicubic
interpolation and SwiGLU. Outputs were finite,FP16 and repeat-bit-identical.
These are operator support checks,not a whole-model precision pass.

The separately [pinned MATH SDPA preflight](<math_preflight.json>) passed text
head128 and vision head72 with FP16 tensor interfaces. MATH is pinned through
PyTorch's documented backend dispatcher;the original HF operator is unchanged.
Original HF RMSNorm reduction and vision rotary computation retain their FP32
islands. Text rotary cos/sin returnFP16,as the HF implementation requires.
CPU Linear's internal accumulation is library-dependent and is **not** inferred
from its input/output dtype.

## Actual full reference

[Reference source](<hf_cpu_fp16_reference78.py>) is staged under a new private
remote run directory and launched once under a9500s supervisor. It:

- loads the official dense Qwen3-VL parameters on CPU,checks the selected
  original parameter hash against the already-verified source,then converts
  parameters deterministically fromBF16 toFP16;
- keeps original rotary buffersFP32 and uses no FP32 Linear adapter,no NPU
  kernels,no quantization/LoRA and no reused oldBF16 teacher features;
- consumes the real2043-token,two-image FP16 pixel/grid fixture;
- recomputes fresh vision,embedding,M-RoPE,DeepStack and all50 text layers,
  selecting the unnormalized layer50 boundary;
- records raw layer boundaries,finite/range checks,Linear dtype observations
  and a pre-finalnorm/layer49 identity check;
- removes saving hooks before a second full run,then requires repeat-bit
  identity at the selected boundary before reporting completion.

The newer HF multimodal token-type IDs are included. Both preflight receipts
and the write-once evidence helper are staged alongside the source. Two fresh
forks reviewed the reference contract. The [original supervisor identity](<run_started.json>)
prevents uncertain work from being replayed;new attempts must use a new run
root. The [bounded status reader](<reference_status78.py>) never launches a model.

The [official HF LFS metadata](<official_text_manifest_receipt78.json>) pins14
checkpoint SHA256s at [the selected official revision](https://huggingface.co/api/models/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/text_encoder?expand=true).
No model payload was downloaded for this metadata check. The
[post-run verifier](<verify_reference78.py>) will compare every used local shard
against those official hashes;reference completion alone does not authorize an
NPU precision pass.

## Next910A work

[Actual installed encoder audit](<npu_fp16_audit78.json>) confirms that merely
setting the old encoder's half flag is insufficient: it changes parameters,
but pixels/output still pass throughBF16 and several compatibility branches
are BF16-specific. A clean FP16 candidate must remove those round trips and
match the original HF dtype/rounding policy. No production encoder was changed
and no NPU candidate was executed in this round.

[State](<STATE.json>) and [latest archived progress](<progress01.json>) record
whether the original CPU reference is still running or has completed. All
wall-clock numbers here are CPU reference cost,**not910A benchmark latency**.
The whole performance goal remains active;full encoder2s and end-to-end FP16
precision acceptance are not claimed.
