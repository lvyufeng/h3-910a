#!/usr/bin/env python3
"""Serial build and bit-identical frozen-leaf proof before full TP8 loading."""
from pathlib import Path
import json

ROOT=Path('/tmp/h3_round86_late01')


def main():
    import torch
    import torch_npu
    import blocked_dynamic_steps83 as row
    import batch_gemm85 as batch
    from runtime5gm83 import Runtime5GM
    from attention_engine85 import AccurateAttention85
    from full_row_fp32_fixed84 import compare,tensor_info
    from types import SimpleNamespace
    torch.npu.set_device(0)
    records=[row.build_kernel(m=2043,k=k,n=5120,out_dir=ROOT/'row_kernels')for k in(1024,3200)]
    engine=AccurateAttention85(SimpleNamespace(_scaled_dot_product_attention=lambda q,k,v:None),Runtime5GM(0),ROOT/'attention_kernels')
    engine.prepare();engine.set_enabled(True)
    capture=torch.load('/tmp/h3_round85_capture01/QKV_rank0.pt',map_location='cpu',weights_only=True)
    with torch.inference_mode():out=engine.forward(*[capture[x].to('npu:0')for x in('q','k','v')])
    measured=json.loads(Path('/tmp/h3_round85_batch01/attention_result85.json').read_text())
    assert tensor_info(out)['sha256']==measured['custom_vs_CPU_Half_MATH']['npu']['sha256']
    result={'scope':__doc__,'row_objects':records,'engine':engine.snapshot(),
            'matches_executed_R85_frozen_leaf_bits':True,
            'leaf_vs_CPU_Half_MATH':compare(out,capture['cpu_half_math_out']),
            'not_full_encoder_precision':True}
    target=ROOT/'prebuild_late86.json';assert not target.exists()
    with target.open('x')as f:f.write(json.dumps(result,indent=2)+'\n')
    print('Bothrowshapesandall4attentionkernelsprecompiled;reusableengineequalsactualfrozenleafbit-for-bit')


if __name__=='__main__':main()
