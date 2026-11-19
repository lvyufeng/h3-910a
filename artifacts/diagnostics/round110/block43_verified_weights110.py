#!/usr/bin/env python3
"""Exact CPU raw42 rebase into NPU block43: genuine original-Half one-block precision diagnostic.

Not a full-encoder benchmark. No injected final norm or DeepStack, no Float weights.
"""
from pathlib import Path
import os
import json
import hashlib
ROOT=Path('/tmp/h3_round110_block43_01')
CPU_CAPTURE=Path('/tmp/h3_round109_cpu02')


def main():
    import torch
    import torch_npu
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    import full_row_fp32_fixed84 as previous
    from evidence_io110 import EvidenceWriter
    from runtime5gm83 import Runtime5GM
    from attention_engine85 import AccurateAttention85
    from kv_padding81 import install as install_kv
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel,destroy_model_parallel
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('block43_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,'source_SHA256':previous.file_sha(Path(__file__)),
       'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None,'cases':{},
       'not_full_encoder_timing':True,'no_FloatLinear_teacher_BFgoldcast_or_fake_final_norm':True}
    ready=False;group=None;kv=None;attention=None
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank);previous.ROOT=ROOT
        reference=previous.load_cpu_gate(r);candidate=previous.load_candidate(r);previous.install_fp16_f32_math_attention(candidate,r)
        kv=install_kv(candidate)
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),local_files_only=True)
        assert cfg.text_config.intermediate_size==25600
        encoder=candidate.MiniMaxH3Qwen3VLEncoder(str(previous.MODEL),device=torch.device('npu',rank),load_model=False,encoder_group=group)
        encoder.compute_dtype=torch.float16;encoder._r79_fp16=True;encoder._tp_size=size
        encoder.text_model=torch.nn.Module()
        encoder.text_model.layers=torch.nn.ModuleDict({'43':candidate.MiniMaxH3Qwen3VLTextDecoderLayer(group,cfg.text_config,torch.float16,prefix='text_encoder.text_model.layers.43')})
        encoder.text_model.rotary_emb=candidate.MiniMaxH3Qwen3VLTextRotaryEmbedding(cfg.text_config)
        idx=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        names=[name for name in idx if name.startswith('model.language_model.layers.43.')]
        assert len(names)==11
        r['originalCPUHalf_full_layer43_weight_SHAs_verified']={}
        def weights():
            for shard in sorted({idx[name]for name in names}):
                with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as h:
                    for name in sorted(names):
                        if idx[name]==shard:
                            value=h.get_tensor(name)
                            key=name.removeprefix('model.')
                            converted_sha=previous.tensor_info(value.to(torch.float16))['sha256']
                            assert converted_sha==reference['parameter_conversion'][key]['reference_sha256']
                            r['originalCPUHalf_full_layer43_weight_SHAs_verified'][key]=converted_sha
                            yield name,value
        with torch.no_grad():loaded=encoder.load_weights(weights())
        assert loaded==set(dict(encoder.named_parameters()))
        assert len(r['originalCPUHalf_full_layer43_weight_SHAs_verified'])==11
        encoder.to(device=torch.device('npu',rank));encoder.eval();assert all(p.dtype==torch.float16 for p in encoder.parameters())
        attention=AccurateAttention85(candidate,Runtime5GM(rank),ROOT/'attention_kernels');attention.prepare();attention.set_enabled(False)
        capture_receipt=json.loads((CPU_CAPTURE/'CPU_raw_reference109.json').read_text())
        assert capture_receipt['completed']and capture_receipt['baseline_passes']['passes']==2
        assert capture_receipt['baseline_passes']['all_25_R78_boundaries_matched']and capture_receipt['baseline_passes']['repeat_bit_identical_all_25']
        assert capture_receipt['captured_pair_replay']['comparison']['raw_bits_equal']
        assert capture_receipt['layers_parameters']['unchanged']and capture_receipt['r78_files_after']['unchanged']
        for i in(42,43):
            assert previous.file_sha(CPU_CAPTURE/('CPU_raw%d_109.pt'%i))==capture_receipt['captured_tensors'][str(i)]['file_sha256']
        r['original_CPU_pair_receipt_SHA256']=previous.file_sha(CPU_CAPTURE/'CPU_raw_reference109.json')
        cpu_input=torch.load(CPU_CAPTURE/'CPU_raw42_109.pt',map_location='cpu',weights_only=True)
        cpu_gold=torch.load(CPU_CAPTURE/'CPU_raw43_109.pt',map_location='cpu',weights_only=True)
        truths={row['layer']:row['output']['sha256']for row in reference['language_layers']}
        assert cpu_input.shape==cpu_gold.shape==(1,2043,5120)and cpu_input.dtype==cpu_gold.dtype==torch.float16
        assert previous.tensor_info(cpu_input)['sha256']==truths[42]and previous.tensor_info(cpu_gold)['sha256']==truths[43]
        cpu_inputs=torch.load(previous.CPU_ROOT/'FP16_multimodal_inputs.pt',map_location='cpu',weights_only=True)
        dev=torch.device('npu',rank);inp=cpu_input.to(dev);positions=cpu_inputs['position_ids'].to(dev)
        assert previous.compare(inp,cpu_input)['raw_bits_equal']
        param_before=previous.parameters_sha(encoder);input_sha=previous.tensor_info(inp)['sha256'];native=None
        capture={}
        def capture_mlp(module,args):capture['mlp_input']=args[0].detach().clone()
        for name,accurate in(('native',False),('precise_attention',True)):
            attention.set_enabled(accurate)
            handle=encoder.text_model.layers['43'].mlp.register_forward_pre_hook(capture_mlp)if name=='native'else None
            try:
                with torch.inference_mode():
                    pos=encoder.text_model.rotary_emb(inp,positions)
                    out=encoder.text_model.layers['43'](inp.clone(),position_embeddings=pos)
                    repeated=encoder.text_model.layers['43'](inp.clone(),position_embeddings=pos)
            finally:
                if handle is not None:handle.remove()
            assert previous.compare(repeated,out)['raw_bits_equal']
            r['cases'][name]={'exact_CPU_raw42_input':previous.tensor_info(cpu_input),'raw43_vs_original_CPU_Half':previous.compare(out,cpu_gold),
                             'repeated_bits_equal':True,'not_full_encoder_time':True,'attention_snapshot':attention.snapshot()}
            if name=='native':native=out.detach().cpu()
        attention.set_enabled(False)
        with torch.inference_mode():restored=encoder.text_model.layers['43'](inp.clone(),position_embeddings=encoder.text_model.rotary_emb(inp,positions))
        r['native_recovery']=previous.compare(restored,native);assert r['native_recovery']['raw_bits_equal']
        from late_leaf86 import run_controls
        # Same-operand CPU Half leaf controls are outside ANY inference benchmark;
        # layer index43, not a duplicate of previous layer25 experiment.
        with torch.inference_mode():r['actual_layer43_MLP_leaf_controls']=run_controls(encoder.text_model.layers['43'],inp,capture['mlp_input'],previous.compare,previous.tensor_info)
        r['original_parameters_unchanged']=param_before==previous.parameters_sha(encoder);assert r['original_parameters_unchanged']
        r['exact_CPU_input_unchanged']=input_sha==previous.tensor_info(inp)['sha256'];assert r['exact_CPU_input_unchanged']
        r['original_CPU_gold_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9';assert r['original_CPU_gold_unchanged']
        r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if attention is not None:attention.uninstall()
        if kv is not None:kv.uninstall()
        if group is not None:group.destroy()
        if ready:destroy_model_parallel();torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
