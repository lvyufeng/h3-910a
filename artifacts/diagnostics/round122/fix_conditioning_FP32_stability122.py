#!/usr/bin/env python3
"""Fix the measured Half output overflow only in H3's conditioning small branch.

Condition projection and refiner residual/RMS arithmetic become FP32. Refiner
attention/MLP and all 50 main DiT blocks remain FP16. Original checkpoint values,
normalization definitions/epsilons, masks, sampler and input are unchanged. No
clipping and no BF16 fallback. This is explicitly a mixed-precision candidate,
not a successful pure-Half original trial.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/mseco/h3_gpu12201')
PATH = ROOT / 'vllm-omni-fl2va-release/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py'


def main():
    receipt = ROOT / 'conditioning_FP32_stability_fix122.json'
    assert not receipt.exists()
    probe = json.loads((ROOT / 'conditioning_range_probe122.json').read_text())
    assert probe['confirmed_Half_output_range_overflow']
    original = PATH.read_text()
    prior = json.loads((ROOT / 'fp16_source_adaptation122b.json').read_text())['files']['minimax_h3_transformer.py']['adapted_sha256']
    assert hashlib.sha256(original.encode()).hexdigest() == prior
    text = original
    substitutions = [
        ('        "video_patch_proj.weight",\n', '        "condition_proj.weight",\n        "condition_proj.bias",\n        "video_patch_proj.weight",\n'),
        ('''        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_BF16_DTYPE,
''', '''        self.condition_proj = ColumnParallelLinear(
            arch.text_dim,
            arch.hidden_size,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
'''),
        ('text_rows = text_embeddings_selected.to(device=device, dtype=_BF16_DTYPE)', 'text_rows = text_embeddings_selected.to(device=device, dtype=_FP32_DTYPE)'),
        ('class MiniMaxH3TokenRefinerBlock(nn.Module):', '''def _conditioning_residual_rms_norm(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    # Preserve the real learned RMSNorm and epsilon. Normalize in FP32
    # before narrowing the bounded result for FP16 attention/MLP matrices.
    return nn.functional.rms_norm(
        x.float(), norm.normalized_shape, norm.weight.float(), norm.eps
    ).to(_BF16_DTYPE)


class MiniMaxH3TokenRefinerBlock(nn.Module):'''),
        ('''        x = x + self.attn(
            self.norm1(x),
            rope_freqs=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x
''', '''        residual = x.float()
        h = self.attn(
            _conditioning_residual_rms_norm(self.norm1, residual),
            rope_freqs=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        x = residual + h.float()
        h = self.mlp(_conditioning_residual_rms_norm(self.norm2, x))
        return x + h.float()
'''),
        ('        return self.final_norm(x)\n', '        return _conditioning_residual_rms_norm(self.final_norm, x)\n'),
    ]
    for old, new in substitutions:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    compile(text, str(PATH), 'exec')
    with (ROOT / 'transformer_original_FP16_before_stability122.py').open('x') as f:
        f.write(original)
    PATH.write_text(text)
    patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True), text.splitlines(keepends=True),
                                       fromfile='original_FP16/minimax_h3_transformer.py', tofile='conditioning_FP32/minimax_h3_transformer.py'))
    (ROOT / 'conditioning_FP32_stability_fix122.patch').write_text(patch)
    result = {'scope': __doc__, 'applied': True, 'mixed_precision': True,
              'original_sha256': prior, 'modified_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'confirmed_original_Half_projection_output_max_in_FP32': probe['FP32_using_identical_FP16_input_and_weight_values']['finite_max_abs'],
              'Half_range_limit': probe['FP16_finite_limit'],
              'new_FP32_parameter_names': ['condition_proj.weight', 'condition_proj.bias'],
              'new_FP32_activation_paths': ['condition_proj output', 'token_refiner residual streams and RMS accumulation'],
              'unchanged_FP16_matrix_paths': ['text/vision encoder', 'token_refiner attention/MLP', 'all 50 main DiT blocks'],
              'checkpoint_bytes_changed': False, 'norm_replaced_with_fake': False, 'clipping_added': False,
              'BF16_fallback': False, 'full_generation_still_required': True}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
