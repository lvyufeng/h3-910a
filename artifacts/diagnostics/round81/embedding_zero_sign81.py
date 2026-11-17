#!/usr/bin/env python3
"""Check whether R80 embedding hash mismatch is solely signed-zero reduction.

Original checkpoint row lookup thenFP16 conversion is independently hash-checked
against the CPU FP16 reference. A localFP32 sum with sevenzero rank contributions
models the vocab-parallel reduction;it is not an actual newHCCLrun or a newteacher.
"""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open
from evidence_io81 import EvidenceWriter

ROOT=Path('/tmp/h3_round81_embedding_sign01')
MODEL=Path('/home/mseco/models/MiniMax-H3/FL2VA/text_encoder')
CPU=Path('/tmp/h3_round78_hf_fp1601')
INPUT=Path('/tmp/h3_round54_capture01/encoder_input.pt')
writer=EvidenceWriter(ROOT/'embedding_sign.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'CPU_reference_changed':False,'production_changed':False,'actual_HCCL_reexecuted':False,'whole_goal_complete':False}


def raw(value):
    return hashlib.sha256(memoryview(value.detach().contiguous().view(torch.uint8).numpy())).hexdigest()


try:
    torch.set_num_threads(4)
    proof=json.loads((CPU/'reference.json').read_text());assert proof['completed']and proof['repeat_pre_final_norm_bits_equal']
    observed=json.loads(Path('/tmp/h3_round80_full_fp16_01/tp8_rank0.json').read_text())['cases']['cpu_fp16_comparison']['embedding']
    inputs=torch.load(INPUT,map_location='cpu',weights_only=True);index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    key='model.language_model.embed_tokens.weight'
    with safe_open(str(MODEL/index[key]),framework='pt',device='cpu')as f:
        embedding=torch.nn.functional.embedding(inputs['input_ids'][None],f.get_tensor(key)).to(torch.float16)
    assert raw(embedding)==proof['embedding_output']['sha256']
    summed=embedding.float();zero=torch.zeros_like(summed)
    for _ in range(7):summed=summed+zero
    simulated=summed.to(torch.float16)
    numerical_equal=bool(torch.equal(simulated,embedding));bits_equal=bool(torch.equal(simulated.view(torch.uint8),embedding.view(torch.uint8)))
    sign_changed=((simulated==0)&(embedding==0)&(simulated.view(torch.int16)!=embedding.view(torch.int16)))
    r.update({'reference_embedding_raw_sha256':raw(embedding),'R80_observed_NPU_embedding_raw_sha256':observed['npu']['sha256'],
              'local_FP32_reduction_model_raw_sha256':raw(simulated),'numeric_equal_before_after':numerical_equal,'raw_bits_equal_before_after':bits_equal,
              'signed_zero_changed_elements':int(sign_changed.sum()),'all_bit_differences_are_signed_zeros':bool(((simulated.view(torch.int16)==embedding.view(torch.int16))|sign_changed).all()),
              'local_model_matches_actual_R80_NPU_embedding_hash':raw(simulated)==observed['npu']['sha256'],
              'FP16_reference_embedding_unchanged':True,'full_encoder_precision_accepted':False,'completed':True})
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
