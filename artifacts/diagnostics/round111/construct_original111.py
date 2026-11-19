#!/usr/bin/env python3
"""Construct only the REAL original-Half condition projection/refiner, without full DiT.

Uses unchanged production classes and the actual native load_weights method; TP1 is the
DiT's replicated pre-SP refiner configuration, not the encoder's TP8 group. No final norm
is added to the encoder: this wrapper calls the model's pre-existing token_refiner norm.
No full encoder benchmark or co-residency qualification is claimed by this module.
"""
from pathlib import Path
import hashlib
import json
import types

CHECKPOINT=Path('/home/mseco/models/MiniMax-H3-native-fp16-audit53/FL2VA/transformer')


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def tensor_info(value):
    import torch
    x=value.detach().cpu().contiguous()
    return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
            'bytes':x.numel()*x.element_size(),'finite':bool(torch.isfinite(x).all())}


def parameter_info(model):
    return {name:tensor_info(p)for name,p in model.named_parameters()}


def compare(left,right):
    import torch
    a=left.detach().cpu().contiguous();b=right.detach().cpu().contiguous()
    assert a.shape==b.shape and a.dtype==b.dtype==torch.float16
    assert bool(torch.isfinite(a).all())and bool(torch.isfinite(b).all())
    aa=a.float().double();bb=b.float().double();rms=((aa-bb).square().mean()).sqrt().item();den=bb.square().mean().sqrt().item()
    return {'npu':tensor_info(a),'cpu':tensor_info(b),'raw_bits_equal':bool(torch.equal(a.view(torch.uint16),b.view(torch.uint16))),
            'max_abs_error':(aa-bb).abs().max().item(),'RMS_error':rms,'relative_rms':rms/max(den,1e-30)}


def verify_sources(pin_path):
    pins=json.loads(Path(pin_path).read_text())
    for name,row in pins.items():assert file_sha(name)==row['sha256'],name
    return {name:row['sha256']for name,row in pins.items()}


def construct_original(pins_path,device):
    import torch
    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_world_size
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as original
    assert get_tensor_model_parallel_world_size()==1
    assert original._BF16_DTYPE==torch.float16
    expected=json.loads(Path(pins_path).read_text());assert expected['all19Halfvalues_exact_original_conversion']
    config=json.loads((CHECKPOINT/'config.json').read_text())
    assert file_sha(CHECKPOINT/'config.json')==expected['checkpoint_config_SHA256']
    assert file_sha(CHECKPOINT/'model.safetensors.index.json')==expected['checkpoint_index_SHA256']
    assert (config['hidden_size'],config['text_dim'],config['num_attention_heads'],config['attention_head_dim'],config['ffn_hidden_size'],config['token_refiner_num_layers'])==(5376,5120,56,128,14336,2)
    assert (config['norm_eps'],config['qk_norm_eps'],config['final_norm_eps'])==(1e-5,1e-5,1e-5)
    arch=original.MiniMaxH3DiTArchConfig.from_mapping(config)

    class ConditionOnlyOriginal(torch.nn.Module):
        def __init__(self):
            super().__init__();self.arch=arch;self.adaln_cache=None;self._diffusers_weights=False
            self.condition_proj=ColumnParallelLinear(arch.text_dim,arch.hidden_size,bias=True,gather_output=True,params_dtype=torch.float16,quant_config=None,prefix='condition_proj')
            self.token_refiner=original.MiniMaxH3TokenRefiner(arch,None,prefix='token_refiner')
        def forward(self,x,cu_seqlens):
            assert x.shape==(2043,5120)and x.dtype==torch.float16
            projected=self.condition_proj(x)[0]
            return self.token_refiner(projected,cu_seqlens=cu_seqlens,max_seqlen=2043,num_requests=1)

    with torch.device('cpu'):model=ConditionOnlyOriginal()
    model.load_weights=types.MethodType(original.MiniMaxH3DiTModel.load_weights,model)
    assert set(dict(model.named_parameters()))==set(expected['parameters'])
    index=json.loads((CHECKPOINT/'model.safetensors.index.json').read_text())['weight_map'];raw={}
    for shard in sorted({index[name]for name in expected['parameters']}):
        with safe_open(str(CHECKPOINT/shard),framework='pt',device='cpu')as f:
            for name,row in expected['parameters'].items():
                if index[name]!=shard:continue
                x=f.get_tensor(name);info=tensor_info(x)
                assert info['dtype']=='torch.float16'and info['shape']==row['shape']and info['sha256']==row['reference_sha256'],name
                raw[name]=x
    loaded=model.load_weights(raw.items());assert loaded==set(expected['parameters'])
    parameters=dict(model.named_parameters())
    for name,value in raw.items():
        if name.endswith('.attn.qkv_proj.weight'):
            ordered=original._reorder_grouped_qkv_to_qkv(value,num_query_groups=56,heads_per_group=1,head_dim=128)
        else:ordered=value
        assert torch.equal(parameters[name].detach().cpu().contiguous().view(torch.uint16),ordered.contiguous().view(torch.uint16)),name
    del raw
    model.eval().to(device);assert all(p.dtype==torch.float16 for p in model.parameters())
    geometry=[]
    for block in model.token_refiner.blocks:
        assert block.attn.num_heads==block.attn.num_kv_heads==56
        assert block.attn.attention.skip_sequence_parallel
        assert not block.attn.attention._has_custom_attention
        geometry.append({'heads':block.attn.num_heads,'head_dim':block.attn.head_dim,'attention_backend':block.attn.attention.attn_backend.get_name(),
                         'skip_sequence_parallel':block.attn.attention.skip_sequence_parallel,'qkv_weight_shape':list(block.attn.qkv_proj.weight.shape),
                         'out_weight_shape':list(block.attn.out_proj.weight.shape),'fc1_weight_shape':list(block.mlp.fc1.weight.shape),
                         'fc2_weight_shape':list(block.mlp.fc2.weight.shape)})
    return model,{'parameters':parameter_info(model),'actual_arch_config':config,'native_loader_and_grouped_QKV_roundtrip':True,
                  'geometry':geometry,'TP_world_size':1,'no_custom_attention_replacement':True,'parameter_bytes':sum(p.numel()*p.element_size()for p in model.parameters())}
