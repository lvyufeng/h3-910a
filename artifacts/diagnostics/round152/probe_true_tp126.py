#!/usr/bin/env python3
"""Verify actual H3 tensor-parallel shards on meta tensors; load no weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tp', type=int, required=True)
    p.add_argument('--usp', type=int, required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    assert root == Path('/tmp/h3_silu152')
    assert '/' not in args.label and args.tp * args.usp == 8
    out = root/args.label
    assert out.is_dir()
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    destination = out/f'rank{rank}.json'
    assert not destination.exists()
    report = {'scope': __doc__, 'completed': False, 'rank': rank, 'tp_requested': args.tp,
              'usp_requested': args.usp, 'weights_loaded': False,
              'forward_executed': False, 'CPU_model_inference': False}
    import torch
    import torch_npu
    from vllm_omni.diffusion.data import OmniDiffusionConfig, DiffusionParallelConfig, AttentionConfig, AttentionSpec, TransformerConfig
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_distributed_env
    from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size, get_tp_group
    from h3_replay123.transformer import MiniMaxH3DiTModel
    torch.npu.set_device(local_rank)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    try:
        init_distributed_environment(world_size=8, rank=rank, local_rank=local_rank,
                                    distributed_init_method='env://', backend='hccl')
        initialize_model_parallel(tensor_parallel_size=args.tp, sequence_parallel_size=args.usp,
                                  ulysses_degree=args.usp, ring_degree=1, backend='hccl')
        report['tp_actual'] = get_tensor_model_parallel_world_size()
        report['tp_group_ranks'] = list(get_tp_group().ranks)
        assert report['tp_actual'] == args.tp
        config_path = Path(args.checkpoint)/'config.json'
        data = config_path.read_bytes()
        arch = json.loads(data)
        report['config_sha256'] = hashlib.sha256(data).hexdigest()
        pc = DiffusionParallelConfig(tensor_parallel_size=args.tp, sequence_parallel_size=args.usp,
                                     ulysses_degree=args.usp, ring_degree=1)
        od = OmniDiffusionConfig(model=str(Path(args.checkpoint).parent), model_class_name='MiniMaxH3Pipeline',
                                 dtype=torch.float16, tf_model_config=TransformerConfig.from_dict(arch),
                                 parallel_config=pc, num_gpus=8,
                                 diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend='TORCH_SDPA')))
        with set_current_diffusion_config(od), set_forward_context(omni_diffusion_config=od), torch.device('meta'):
            model = MiniMaxH3DiTModel(od, quant_config=None)
        report['per_rank_parameter_elements'] = sum(p.numel() for p in model.parameters())
        report['per_rank_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
        report['parameter_elements_by_dtype'] = {
            str(dtype):sum(p.numel() for p in model.parameters() if p.dtype == dtype)
            for dtype in {p.dtype for p in model.parameters()}}
        report['selected_parameter_shapes'] = {
            name:{'shape':list(param.shape),'dtype':str(param.dtype),'device':str(param.device),
                  'class':type(param).__name__}
            for name,param in model.named_parameters()
            if name.startswith(('blocks.0.', 'token_refiner.blocks.0.', 'video_patch_proj.',
                                'audio_patch_proj.', 'condition_proj.', 'time_embedder.', 'final_layer.'))}
        report['local_q_heads'] = model.blocks[0].attn.num_heads
        report['local_kv_heads'] = model.blocks[0].attn.num_kv_heads
        assert report['local_q_heads'] * args.tp == arch['num_attention_heads']
        report['completed'] = True
        destination.write_text(json.dumps(report, indent=2)+'\n')
        if rank == 0:
            print(json.dumps({k:v for k,v in report.items() if k!='selected_parameter_shapes'},indent=2),flush=True)
    except BaseException as exc:
        report['failure'] = {'type':type(exc).__name__,'message':str(exc)}
        destination.write_text(json.dumps(report, indent=2)+'\n')
        raise
    finally:
        if torch.distributed.is_initialized():
            destroy_distributed_env()

if __name__ == '__main__':
    main()
