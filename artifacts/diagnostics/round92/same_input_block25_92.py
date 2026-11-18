#!/usr/bin/env python3
"""Same realHalf input NPU block25 vs originalCPUHalf block25;not encoder timing."""
from pathlib import Path
import json
import os
ROOT=Path(os.environ.get('R92_CASE_ROOT','/tmp/h3_round92_case01'))


def main():
    import torch
    import torch_npu
    import full_row_fp32_fixed84 as previous
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from evidence_io92 import EvidenceWriter
    rank=int(os.environ['RANK']);size=int(os.environ['WORLD_SIZE']);assert size==8
    writer=EvidenceWriter(ROOT/('block25_rank%d.json'%rank))
    r={'scope':__doc__,'rank':rank,'world_size':size,'completed':False,
       'source_SHA256':previous.file_sha(Path(__file__)),'cases':{},'not_full_encoder_timing':True,
       'no_DeepStack_or_final_norm_inserted':True,'full_encoder_precision_accepted':False,'whole_goal_complete':False,'numerical_tolerance':None}
    kv=None;row=None;engine=None;group=None;ready=False
    try:
        torch.set_num_threads(2);torch.npu.set_device(rank);previous.ROOT=ROOT
        proof=previous.load_cpu_gate(r);candidate=previous.load_candidate(r);previous.install_fp16_f32_math_attention(candidate,r)
        from kv_padding81 import install as install_kv
        kv=install_kv(candidate)
        from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator
        from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment,initialize_model_parallel
        init_distributed_environment(world_size=size,rank=rank,local_rank=rank,backend='hccl');ready=True
        initialize_model_parallel(sequence_parallel_size=size,ulysses_degree=size,tensor_parallel_size=1,backend='hccl')
        group=GroupCoordinator([list(range(size))],rank,'hccl')
        cfg=Qwen3VLConfig.from_pretrained(str(previous.MODEL),trust_remote_code=False,local_files_only=True)
        assert cfg.text_config.intermediate_size==25600
        encoder=candidate.MiniMaxH3Qwen3VLEncoder(str(previous.MODEL),device=torch.device('npu',rank),load_model=False,encoder_group=group)
        encoder.compute_dtype=torch.float16;encoder._r79_fp16=True;encoder._tp_size=size
        encoder.text_model=torch.nn.Module()
        encoder.text_model.layers=torch.nn.ModuleDict({'25':candidate.MiniMaxH3Qwen3VLTextDecoderLayer(group,cfg.text_config,torch.float16,prefix='text_encoder.text_model.layers.25')})
        encoder.text_model.rotary_emb=candidate.MiniMaxH3Qwen3VLTextRotaryEmbedding(cfg.text_config)
        idx=json.loads((previous.MODEL/'model.safetensors.index.json').read_text())['weight_map']
        names=[name for name in idx if name.startswith('model.language_model.layers.25.')]
        def tensors():
            for shard in sorted({idx[name]for name in names}):
                with safe_open(str(previous.MODEL/shard),framework='pt',device='cpu')as h:
                    for name in sorted(names):
                        if idx[name]==shard:yield name,h.get_tensor(name)
        loaded=encoder.load_weights(tensors());assert loaded==set(dict(encoder.named_parameters()))
        encoder.to(torch.device('npu',rank));encoder.eval();assert all(p.dtype==torch.float16 for p in encoder.parameters())
        from types import SimpleNamespace
        kv.prime(SimpleNamespace(layers=list(encoder.text_model.layers.values())))
        from runtime5gm83 import Runtime5GM
        from row_fp32_adapter_fixed84 import install as install_row
        from attention_engine85 import AccurateAttention85
        row=install_row(candidate,Runtime5GM(rank),ROOT/'row_kernels',expected_layer_count=2)
        row.prime(encoder.text_model);row.set_enabled(False)
        engine=AccurateAttention85(candidate,Runtime5GM(rank),ROOT/'attention_kernels');engine.prepare();engine.set_enabled(False)
        cpu_receipt=json.loads((ROOT/'CPU_block25_reference92.json').read_text())
        assert cpu_receipt['completed']
        outputs_file=ROOT/'CPU_block25_outputs92.pt'
        assert previous.file_sha(outputs_file)==cpu_receipt['outputs_file_SHA256']
        cpu_outputs=torch.load(outputs_file,map_location='cpu',weights_only=True)['outputs']
        assert previous.tensor_info(cpu_outputs['CPU_raw24'])['sha256']==proof['language_layers'][25]['output']['sha256']
        bundle=torch.load(ROOT/'NPU_raw24_inputs92.pt',map_location='cpu',weights_only=True)
        cpu_inputs={'CPU_raw24':torch.load(previous.CPU_ROOT/'FP16_layer24.pt',map_location='cpu',weights_only=True),**bundle['inputs']}
        assert previous.file_sha(previous.CPU_ROOT/'FP16_layer24.pt')==previous.EXPECTED_CPU_LAYERS[24]
        for name,value in bundle['inputs'].items():assert previous.tensor_info(value)['sha256']==bundle['source_raw24_SHAs'][name]
        before=previous.parameters_sha(encoder);positions=bundle['position_ids'].to(torch.device('npu',rank))
        native_by_input={};saved={}
        layer=encoder.text_model.layers['25']
        for input_case,cpu_input in cpu_inputs.items():
            value=cpu_input.to(torch.device('npu',rank));value_sha=previous.tensor_info(value)['sha256']
            assert previous.compare(value,cpu_input)['raw_bits_equal']
            def forward():
                pos=encoder.text_model.rotary_emb(value,positions)
                return layer(value.clone(),position_embeddings=pos).cpu()
            r['cases'][input_case]={}
            for mode,attn,f32row in(('native',False,False),('precise_attention_native_rows',True,False),
                                   ('native_attention_FP32_rows',False,True),('precise_attention_FP32_rows',True,True)):
                engine.set_enabled(attn);row.set_enabled(f32row)
                with torch.inference_mode():out=forward();repeat=forward()
                repeat_metrics=previous.compare(repeat,out);assert repeat_metrics['raw_bits_equal']
                vs_same_CPU=previous.compare(out,cpu_outputs[input_case]);assert vs_same_CPU['finite']
                r['cases'][input_case][mode]={'same_input_vs_original_CPU_Half_block25':vs_same_CPU,
                                             'vs_original_CPU_layer25':previous.compare(out,cpu_outputs['CPU_raw24']),
                                             'repeat_bit_identical':True,'original_input_SHA256':value_sha,
                                             'engine':engine.snapshot(),'row_adapter':row.snapshot()}
                if input_case=='CPU_raw24':
                    known={'native':'eb7b6ecaa0b34f0a9264f6838385729761b1124c6bad65cfbda1a30b17b1cc62',
                           'precise_attention_native_rows':'9fe39c38dacf6849717723e080eaa40ee5d5ff4e750ec603e65381e1281f24df',
                           'precise_attention_FP32_rows':'7e8056974399832b953f881ed14c7e1316e99e37dfa81a8747a5c7f0e68bce9a'}
                    if mode in known:assert previous.tensor_info(out)['sha256']==known[mode],'oneblockdoesnotreproduceR86raw25'
                if mode=='native':native_by_input[input_case]=out
                saved[(input_case,mode)]=out
            engine.set_enabled(False);row.set_enabled(False)
            with torch.inference_mode():restore=forward()
            assert previous.compare(restore,native_by_input[input_case])['raw_bits_equal']
            assert value_sha==previous.tensor_info(value)['sha256']
            r['cases'][input_case]['native_recovery_and_input_SHA_unchanged']=True;writer.write(r)
        r['original_parameters_unchanged']=before==previous.parameters_sha(encoder);assert r['original_parameters_unchanged']
        r['CPU_reference_SHA_unchanged']=previous.file_sha(previous.CPU_ROOT/'reference.json')=='44acac61f9b4d9bc8afc91632392502ccaa29dee7bfab951e2b15902a0be74c9'
        assert r['CPU_reference_SHA_unchanged']
        if rank==0:
            target=ROOT/'NPU_block25_outputs92.pt';assert not target.exists();torch.save(saved,target)
            r['NPU_output_file_SHA256']=previous.file_sha(target)
        r['completed']=True
    except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        if engine is not None:engine.uninstall()
        if row is not None:row.uninstall()
        if kv is not None:kv.uninstall()
        if group is not None:group.destroy()
        if ready:
            from vllm_omni.diffusion.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
            if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        writer.write(r)


if __name__=='__main__':main()
