#!/usr/bin/env python3
"""Preserve FP32 gated residuals across H3 main blocks after observed Half overflow.

The second real vLLM-Omni trial passed conditioning and both refiner blocks but
block0 returned one +Inf. This candidate avoids narrowing gated residual sums to
Half and normalizes in FP32 before the bounded Half matrix-input boundary. Matrix
weights/inputs in all 50 main DiT attention/MLP/AdaLN remain FP16. No clipping,
BF16 fallback, fake normalization, dropped keyframes or checkpoint changes.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
PATH = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'


def main():
    result_file = ROOT / 'main_residual_FP32_stability_fix122.json'
    assert not result_file.exists()
    previous = json.loads((ROOT / 'conditioning_FP32_stability_fix122.json').read_text())
    original = PATH.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == previous['modified_sha256']
    old_gate = '''    # Apply the per-index gated residual: x + gate[idx] * other.
    return (x + gate.index_select(0, indices) * other).to(dtype)
'''
    new_gate = '''    # Keep gated residual accumulation in FP32. A real first-step
    # block0 result exceeded the Half range; narrowing this state before
    # the following RMSNorm creates Inf even when all matmuls are finite.
    del dtype
    return x.float() + gate.index_select(0, indices).float() * other.float()
'''
    old_norm = '''    return nn.RMSNorm(size, eps=eps, dtype=dtype)
'''
    new_norm = '''    return _MiniMaxH3FP32AccumRMSNorm(size, eps=eps, dtype=dtype)


class _MiniMaxH3FP32AccumRMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is the real learned RMSNorm with its original epsilon.
        # Preserve the incoming stream dtype after FP32 accumulation.
        # This retains FP32 residual streams and keeps Q/K norms Half,
        # matching the FP16 attention Q/K/V dtype contract.
        return nn.functional.rms_norm(
            x.float(), self.normalized_shape, self.weight.float(), self.eps
        ).to(x.dtype)
'''
    assert original.count(old_gate) == 1 and original.count(old_norm) == 1
    changed = original.replace(old_gate, new_gate).replace(old_norm, new_norm)
    compile(changed, str(PATH), 'exec')
    with (ROOT / 'transformer_conditioning_only_fix_before_main_residual122.py').open('x') as f:
        f.write(original)
    PATH.write_text(changed)
    (ROOT / 'main_residual_FP32_stability_fix122.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='conditioning_only/minimax_h3_transformer.py', tofile='FP32_residual/minimax_h3_transformer.py')))
    r = {'scope': __doc__, 'applied': True, 'original_sha256': previous['modified_sha256'],
         'modified_sha256': hashlib.sha256(changed.encode()).hexdigest(),
         'main_DiT_matrix_weights_and_inputs_FP16': True,
         'main_DiT_gated_residual_and_RMS_accumulation_FP32': True,
         'existing_conditioning_stability_fix_retained': True,
         'new_FP32_parameters_added_by_this_fix': [], 'BF16_fallback': False,
         'norm_formula_and_epsilon_unchanged': True, 'clipping_added': False,
         'full_generation_still_required': True}
    result_file.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
