#!/usr/bin/env python3
"""910A same-input FP32 patch-projection leaf check against actual GPU output.

Retain original FP32 projection weights. Compare native FP32 F.linear with a
four-GEMM residual decomposition: both operands of every GEMM are FP16, outputs
and sums are FP32. Residual halves use an exact 2^12 scale to avoid small Half
underflow. This is a bounded 64-row actual-input leaf test, not full inference,
not a CPU baseline, and not a claim of bit-exact FP32 or BF16 parity.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/tmp/h3_align12301')


def main():
    import torch
    import torch_npu
    from safetensors import safe_open
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.set_num_threads(4)
    output = ROOT / 'NPU_FP32_patch_residual_leaf123.json'
    assert not output.exists()
    bundle = torch.load(ROOT / 'same_initial_sampling123.pt', map_location='cpu', weights_only=True)
    checkpoint = Path('/home/mseco/models/MiniMax-H3/FL2VA/transformer')
    index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
    result = {'scope': __doc__, 'completed': False, 'no_local_model_weights': True,
              'CPU_inference_executed': False, 'HF32_enabled': False, 'numeric_tolerance': None,
              'full_model_or_speed_accepted': False, 'cases': {}}
    def metric(a, b):
        a=a.double();b=b.double();difference=a-b
        return {'relative_RMS':float(difference.square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-30)),
                'max_abs_error':float(difference.abs().max()), 'finite':bool(torch.isfinite(a).all()),
                'GPU_max_abs':float(b.abs().max()), 'NPU_max_abs':float(a.abs().max())}
    for name, key in [('video_patch_proj', 'initial_video_rows_with_keyframes'), ('audio_patch_proj', 'initial_audio_rows')]:
        reference_path = ROOT / f'GPU_{name}_sample_rank0.pt'
        reference = torch.load(reference_path, map_location='cpu', weights_only=True)
        row_indices = reference['row_indices']
        x_cpu = bundle[key].index_select(0, row_indices).float().contiguous()
        weights = {}
        for suffix in ['weight', 'bias']:
            param=name+'.'+suffix
            with safe_open(str(checkpoint / index[param]), framework='pt', device='cpu') as file:
                weights[suffix]=file.get_tensor(param).float().contiguous()
        x=x_cpu.to('npu');w=weights['weight'].to('npu');bias=weights['bias'].to('npu')
        gpu=reference['value']
        report={'shape':[len(x),x.shape[1],w.shape[0]],
                'input_raw_sha256':hashlib.sha256(x_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
                'weight_original_FP32_sha256':hashlib.sha256(weights['weight'].view(torch.uint8).numpy().tobytes()).hexdigest(),
                'GPU_reference_sample_file_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(), 'methods':{}}
        with torch.inference_mode():
            torch.npu.synchronize();start=time.perf_counter()
            native=torch.nn.functional.linear(x,w,bias)
            torch.npu.synchronize()
            report['methods']['native_FP32_projection']={**metric(native.cpu(),gpu),'leaf_elapsed_s_not_benchmark':time.perf_counter()-start}
            xh=x.half();wh=w.half()
            scale=4096.0
            xl=((x-xh.float())*scale).half();wl=((w-wh.float())*scale).half()
            assert all(bool(torch.isfinite(t).all()) for t in [xh,wh,xl,wl])
            torch.npu.synchronize();start=time.perf_counter()
            y=torch.mm(xh,wh.t().contiguous(),out_dtype=torch.float32)
            y.add_(torch.mm(xh,wl.t().contiguous(),out_dtype=torch.float32), alpha=1.0/scale)
            y.add_(torch.mm(xl,wh.t().contiguous(),out_dtype=torch.float32), alpha=1.0/scale)
            torch.npu.synchronize()
            report['methods']['three_Half_GEMMs_FP32_output']={**metric((y+bias).cpu(),gpu),'leaf_elapsed_s_not_benchmark':time.perf_counter()-start}
            torch.npu.synchronize();start=time.perf_counter()
            yy=y.clone()
            yy.add_(torch.mm(xl,wl.t().contiguous(),out_dtype=torch.float32),alpha=1.0/(scale*scale))
            yy.add_(bias)
            torch.npu.synchronize()
            report['methods']['four_Half_GEMMs_FP32_output']={**metric(yy.cpu(),gpu),'final_extra_term_elapsed_s_not_benchmark':time.perf_counter()-start}
        result['cases'][name]=report
        output.write_text(json.dumps(result,indent=2)+'\n')
    result['completed']=True
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
