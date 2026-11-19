#!/usr/bin/env python3
"""Range-safe H3 MLP with FP16-input/weight GEMM and native FP32 output.

Measured real vLLM-Omni block18 MLP output overflow motivates FP32 GEMM output and
FP32 TP reduction. FP32 SiLU/product with an exact power-of-two scale prevents the
intermediate-product Half range boundary. The actual down-projection matrix inputs
and weights stay FP16; CUDA torch.mm(..., out_dtype=float32) retains a wide output.
The scale is undone BEFORE each rank's TP all-reduce, so independent partition
scales preserve the mathematical linear transform. No clipping or BF16 fallback.
This changes rounding boundaries and must not be called bitwise/lossless BF16 parity.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
PATH = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'

CLASS = '''class _MiniMaxH3HalfInputFP32OutputLinearMethod(UnquantizedLinearMethod):
    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.device.type != "cuda" or layer.weight.dtype != torch.float16:
            raise TypeError("This trial requires CUDA FP16 matrix weights")
        # The FP32 activation product is narrowed only after a finite
        # power-of-two scale. This is range protection, not clipping.
        product = x.float()
        peak = float(product.abs().amax())
        if not math.isfinite(peak):
            raise FloatingPointError("Nonfinite MLP product before matrix input")
        exponent = max(0, math.ceil(math.log2(peak / 32752.0))) if peak > 32752.0 else 0
        scale = float(2 ** exponent)
        half_input = (product / scale).to(torch.float16)
        result = torch.mm(
            half_input.reshape(-1, half_input.shape[-1]),
            layer.weight.t(), out_dtype=torch.float32,
        ).reshape(*x.shape[:-1], layer.weight.shape[0])
        result = result * scale
        if bias is not None:
            result = result + bias.float()
        return result


'''


def main():
    receipt = ROOT / 'MLP_Half_input_FP32_output_fix122.json'
    assert not receipt.exists()
    prior = json.loads((ROOT / 'main_residual_FP32_stability_fix122.json').read_text())
    original = PATH.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == prior['modified_sha256']
    preflight = json.loads((ROOT / 'Half_input_FP32_output_GEMM_preflight122.json').read_text())
    assert preflight['CUDA_GEMM_FP16_inputs_FP32_output_supported'] and preflight['wide_output_finite']
    substitutions = [
        ('    RowParallelLinear,\n', '    RowParallelLinear,\n    UnquantizedLinearMethod,\n'),
        ('class MiniMaxH3MLP(nn.Module):', CLASS + 'class MiniMaxH3MLP(nn.Module):'),
        ('''            prefix=f"{prefix}.fc2",
        )
''', '''            prefix=f"{prefix}.fc2",
        )
        if quant_config is not None:
            raise ValueError("The reference trial disallows quantization")
        self.fc2.quant_method = _MiniMaxH3HalfInputFP32OutputLinearMethod()
'''),
        ('        hidden = nn.functional.silu(gate) * up\n', '        hidden = nn.functional.silu(gate.float()) * up.float()\n'),
    ]
    changed = original
    for old, new in substitutions:
        assert changed.count(old) == 1, old
        changed = changed.replace(old, new)
    compile(changed, str(PATH), 'exec')
    with (ROOT / 'transformer_residual_only_before_wide_MLP122.py').open('x') as f:
        f.write(original)
    PATH.write_text(changed)
    (ROOT / 'MLP_Half_input_FP32_output_fix122.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='FP32_residual/minimax_h3_transformer.py', tofile='FP16_GEMM_FP32_output/minimax_h3_transformer.py')))
    result = {'scope': __doc__, 'applied': True, 'original_sha256': prior['modified_sha256'],
              'modified_sha256': hashlib.sha256(changed.encode()).hexdigest(),
              'MLP_weight_dtype': 'FP16', 'MLP_down_projection_GEMM_input_dtype': 'FP16',
              'MLP_down_projection_GEMM_output_dtype': 'FP32', 'MLP_TP_reduction_dtype': 'FP32',
              'SwiGLU_point_math': 'FP32 SiLU and product; exact power-of-two range scale before FP16 input; undo scale before TP reduction',
              'new_FP32_matrix_weights_added': [], 'clipping_added': False, 'BF16_fallback': False,
              'checkpoint_bytes_changed': False, 'bit_exact_or_lossless_claim': False,
              'complete_generation_still_required': True}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
