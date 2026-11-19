#!/usr/bin/env python3
"""GPU-only range probe of the exact captured H3 FP16 conditioning projection input.

Not a BF16 reference and not a CPU inference baseline. Reuse the actual first-trial
FP16 encoder output, load the same official conditioning weights directly on this
RTX4090, and distinguish Half-output range overflow from nonfinite input/weights.
"""
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open

ROOT = Path('/home/mseco/h3_gpu12201')


def stats(t):
    x = t.detach()
    finite = torch.isfinite(x)
    selected = x[finite].float()
    return {'dtype': str(x.dtype), 'device': str(x.device), 'shape': list(x.shape),
            'nan': int(torch.isnan(x).sum()), 'positive_inf': int(torch.isposinf(x).sum()),
            'negative_inf': int(torch.isneginf(x).sum()),
            'finite_max_abs': float(selected.abs().max()) if selected.numel() else None,
            'finite_rms': float(selected.square().mean().sqrt()) if selected.numel() else None}


def main():
    report_path = ROOT / 'conditioning_range_probe122.json'
    assert not report_path.exists()
    assert 'RTX 4090' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    captured = torch.load(ROOT / 'results/raw_fp16/vllm_fp16_encoder_hidden122.pt', map_location='cpu', weights_only=True)
    hidden = captured['hidden_states'].cuda()
    tags = captured['token_tags'].cuda()
    checkpoint = ROOT / 'checkpoint/FL2VA/transformer'
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    values = {}
    for name in ['condition_proj.weight', 'condition_proj.bias', 'token_refiner.final_norm.weight']:
        with safe_open(str(checkpoint / index[name]), framework='pt', device='cpu') as f:
            values[name] = f.get_tensor(name).to('cuda')
    half_weight = values['condition_proj.weight'].half()
    half_bias = values['condition_proj.bias'].half()
    r = {'scope': __doc__, 'completed': False, 'input': stats(hidden),
         'weight': stats(half_weight), 'bias': stats(half_bias),
         'original_weight_dtype': str(values['condition_proj.weight'].dtype),
         'token_tag_counts': {str(int(tag)): int((tags == tag).sum()) for tag in torch.unique(tags)},
         'weights_are_official_original_checkpoint': True, 'no_CPU_inference_baseline': True}
    with torch.inference_mode():
        half_output = torch.nn.functional.linear(hidden, half_weight, half_bias)
        fp32_same_half_values = torch.nn.functional.linear(hidden.float(), half_weight.float(), half_bias.float())
        fp32_original_values = torch.nn.functional.linear(hidden.float(), values['condition_proj.weight'].float(), values['condition_proj.bias'].float())
    r['FP16_output'] = stats(half_output)
    r['FP32_using_identical_FP16_input_and_weight_values'] = stats(fp32_same_half_values)
    r['FP32_using_original_weight_values_and_same_input'] = stats(fp32_original_values)
    limit = torch.finfo(torch.float16).max
    overflow = fp32_same_half_values.abs() > limit
    r['FP16_finite_limit'] = float(limit)
    r['FP32_entries_outside_FP16_range'] = int(overflow.sum())
    r['rows_with_out_of_FP16_range_entries'] = int(overflow.any(dim=-1).sum())
    r['overflow_rows_by_token_tag'] = {str(int(tag)): int((overflow.any(dim=-1) & (tags == tag)).sum()) for tag in torch.unique(tags)}
    r['refiner_final_norm_weight'] = stats(values['token_refiner.final_norm.weight'])
    r['completed'] = True
    r['confirmed_Half_output_range_overflow'] = r['FP16_output']['positive_inf'] + r['FP16_output']['negative_inf'] > 0 and r['FP32_using_identical_FP16_input_and_weight_values']['positive_inf'] == 0 and r['FP32_entries_outside_FP16_range'] > 0
    report_path.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2), flush=True)


if __name__ == '__main__':
    main()
