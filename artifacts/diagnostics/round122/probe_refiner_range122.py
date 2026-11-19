#!/usr/bin/env python3
"""Short GPU-only range check of the two conditioning refiner blocks.

Uses the exact captured FP16 encoder output and original checkpoint values. This
is a functional range preflight, NOT a complete vLLM-Omni generation or a BF16
quality comparison. Candidate: FP32 condition projection and FP32 refiner residual/
RMS accumulation, with refiner attention/MLP matrix operations still FP16.
"""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from probe_conditioning_range122 import stats

ROOT = Path('/home/mseco/h3_gpu12201')


def main():
    output = ROOT / 'refiner_range_preflight122.json'
    assert not output.exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cp = ROOT / 'checkpoint/FL2VA/transformer'
    index = json.loads((cp / 'model.safetensors.index.json').read_text())['weight_map']
    hidden = torch.load(ROOT / 'results/raw_fp16/vllm_fp16_encoder_hidden122.pt', map_location='cpu', weights_only=True)['hidden_states'].cuda()
    cache = {}
    def weight(name, dtype=torch.float16):
        key = (name, dtype)
        if key not in cache:
            with safe_open(str(cp / index[name]), framework='pt', device='cpu') as f:
                cache[key] = f.get_tensor(name).to(device='cuda', dtype=dtype)
        return cache[key]
    def norm_fp32_to_half(x, name):
        gamma = weight(name).float()
        return torch.nn.functional.rms_norm(x.float(), (gamma.numel(),), gamma, 1e-5).half()
    r = {'scope': __doc__, 'completed': False, 'finite': False, 'stages': {}}
    def check(name, x):
        s = stats(x)
        r['stages'][name] = s
        output.write_text(json.dumps(r, indent=2) + '\n')
        if s['nan'] or s['positive_inf'] or s['negative_inf']:
            raise FloatingPointError('Range failure: ' + name)
        return x
    with torch.inference_mode():
        x = check('FP32_condition_projection', torch.nn.functional.linear(hidden.float(), weight('condition_proj.weight', torch.float32), weight('condition_proj.bias', torch.float32)))
        for i in range(2):
            p = f'token_refiner.blocks.{i}.'
            residual = x.float()
            h = norm_fp32_to_half(residual, p + 'norm1.weight')
            qkv = check(p + 'FP16_qkv', torch.nn.functional.linear(h, weight(p + 'attn.qkv_proj.weight')))
            q, k, v = qkv.reshape(x.shape[0], 56, 3 * 128).split(128, dim=-1)
            q = torch.nn.functional.rms_norm(q, (128,), weight(p + 'attn.q_norm.weight'), 1e-5)
            k = torch.nn.functional.rms_norm(k, (128,), weight(p + 'attn.k_norm.weight'), 1e-5)
            attention = torch.nn.functional.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], scale=128**-0.5)
            attention = attention[0].transpose(0, 1).reshape(x.shape[0], 56 * 128)
            x = check(p + 'FP32_attention_residual', residual + torch.nn.functional.linear(attention, weight(p + 'attn.out_proj.weight')).float())
            h = norm_fp32_to_half(x, p + 'norm2.weight')
            gate, up = torch.nn.functional.linear(h, weight(p + 'mlp.fc1.weight')).chunk(2, dim=-1)
            product = check(p + 'FP16_SwiGLU_product', torch.nn.functional.silu(gate) * up)
            x = check(p + 'FP32_MLP_residual', x + torch.nn.functional.linear(product, weight(p + 'mlp.fc2.weight')).float())
        x = check('normalized_FP16_refiner_output', norm_fp32_to_half(x, 'token_refiner.final_norm.weight'))
    r['completed'] = True
    r['finite'] = True
    output.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
