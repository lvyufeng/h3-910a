"""Install the accepted original components only; component RPCs do not sample.

No request-scoped encoder save wrapper, DiT replay, final-AdaLN candidate, noise
substitution, video decode or pipeline.forward wrapper is installed here.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

FIXTURE=Path('/tmp/h3_align12401')


def install():
    import torch
    import torch_npu
    from vllm.utils.torch_utils import set_default_torch_dtype
    from h3_replay123.transformer import MiniMaxH3DiTModel,MiniMaxH3Attention
    from fp32_patch_residual_guard123 import install_patch_projection_guard123
    from fp32_time_residual_guard123 import install_time_projection_guard123
    import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    import vllm_omni.diffusion.models.minimax_h3.vae as vae
    torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
    assert os.environ.get('VLLM_OMNI_910A_SHARED_STREAM')=='1'
    for name in ('H3_R126_VAE_OPERAND_CACHE','VLLM_OMNI_910A_FAST_RESIDUAL','VLLM_OMNI_910A_FAST_ACT'):
        assert os.environ.get(name,'0')=='0'
    original_factory=vae._load_remote_component
    def original_FP32_factory(*args,**kwargs):
        with set_default_torch_dtype(torch.float32):result=original_factory(*args,**kwargs)
        assert all(p.dtype==torch.float32 for p in result.parameters() if p.is_floating_point())
        return result
    vae._load_remote_component=original_FP32_factory
    def preserve_original_parameters(decoder,*,device):
        assert all(p.dtype==torch.float32 for p in decoder.parameters() if p.is_floating_point());return False
    vae.install_h3_vae_optimizations=preserve_original_parameters
    path=FIXTURE/'NPU_encoder_FP16_124.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='0959895b778fc0900d785250b71279055d3940d975f3523909856072960f5026'
    spec=importlib.util.spec_from_file_location('h3_original_encoder137',path);encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    pipeline.MiniMaxH3Qwen3VLEncoder=encoder.MiniMaxH3Qwen3VLEncoder
    class OriginalGuardedDiT137(MiniMaxH3DiTModel):
        def __init__(self,od_config,quant_config=None,*,diffusers_weights=False):
            assert quant_config is None and not diffusers_weights
            super().__init__(od_config,quant_config=None);self.diffusers_weights=False
            self._precision_modules137=install_patch_projection_guard123(self)+install_time_projection_guard123(self)
            assert len(self.blocks)==50
        def forward(self,**kwargs):
            raise RuntimeError('R137 component-only diagnostic must never execute DiT or a sampler')
    pipeline.MiniMaxH3DiTModel=OriginalGuardedDiT137;pipeline.MiniMaxH3Attention=MiniMaxH3Attention
    from component_rpc137 import install_rpc
    install_rpc()
    if os.environ.get('H3_R127_INIT_DIAGNOSTICS')=='1':
        from init_diagnostics127 import install as install_init_diagnostics
        install_init_diagnostics()
