#!/usr/bin/env python3
"""Extend measured FP16-GEMM/FP32-output protection to attention output projections.

TP2/USP4 trial first-step block44 attention output overflowed on a subset of row
shards. Keep attention output projection matrix weights/inputs FP16; retain CUDA
FP32 GEMM outputs and FP32 row-parallel reduction, then FP32 gated residuals. This
uses the already tested exact scale/wide-output method, with no clipping, BF16 or
new FP32 matrix-weight copy. Existing precision changes remain explicit.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
PATH = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'


def main():
    destination = ROOT / 'attention_Half_input_FP32_output_fix122.json'
    assert not destination.exists()
    prior = json.loads((ROOT / 'MLP_Half_input_FP32_output_fix122.json').read_text())
    original = PATH.read_text()
    assert hashlib.sha256(original.encode()).hexdigest() == prior['modified_sha256']
    old = '''            prefix=f"{prefix}.out_proj",
        )
'''
    new = '''            prefix=f"{prefix}.out_proj",
        )
        if quant_config is not None:
            raise ValueError("The FP16 reference trial disallows quantization")
        self.out_proj.quant_method = _MiniMaxH3HalfInputFP32OutputLinearMethod()
'''
    assert original.count(old) == 1
    changed = original.replace(old, new)
    compile(changed, str(PATH), 'exec')
    with (ROOT / 'transformer_MLP_only_wide_output_before_attention122.py').open('x') as f:
        f.write(original)
    PATH.write_text(changed)
    (ROOT / 'attention_Half_input_FP32_output_fix122.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile='MLP_wide_output/minimax_h3_transformer.py', tofile='attention_wide_output/minimax_h3_transformer.py')))
    row = {'scope': __doc__, 'applied': True, 'original_sha256': prior['modified_sha256'],
           'modified_sha256': hashlib.sha256(changed.encode()).hexdigest(),
           'attention_out_projection_weight_and_input_FP16': True,
           'attention_out_projection_GEMM_output_and_TP_reduction_FP32': True,
           'new_FP32_matrix_weights_added': [], 'checkpoint_bytes_changed': False,
           'clipping_added': False, 'BF16_fallback': False, 'complete_generation_still_required': True}
    destination.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
