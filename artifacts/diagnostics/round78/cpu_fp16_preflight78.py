#!/usr/bin/env python3
"""Actual CPU FP16 operator coverage preflight;not model accuracy or NPU performance."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
import signal
import time
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
from evidence_io78 import EvidenceWriter

ROOT=Path('/tmp/h3_round78_preflight01')
writer=EvidenceWriter(ROOT/'preflight.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operators':{},
   'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version()},
   'device':'cpu','native_BF16_not_required':True,'FP32_linear_teacher_adapter_used':False,'cpu_reference_generated':False,
   'NPU_precision_accepted':False,'whole_goal_complete':False}


def tensor_info(value):
    assert value.device.type=='cpu'
    return {'shape':list(value.shape),'dtype':str(value.dtype),'finite':bool(torch.isfinite(value.float()).all()),
            'sha256':hashlib.sha256(memoryview(value.detach().contiguous().view(torch.uint8).numpy())).hexdigest()}


def probe(name,function,args,expected=torch.float16):
    row={'inputs':[tensor_info(arg)for arg in args if isinstance(arg,torch.Tensor)],'expected_result_dtype':str(expected)}
    started=time.perf_counter()
    try:
        with torch.inference_mode():one=function(*args);two=function(*args)
        row['seconds_excluded_performance']=time.perf_counter()-started;row['output']=tensor_info(one)
        row['repeat_bits_equal']=torch.equal(one.contiguous().view(torch.uint8),two.contiguous().view(torch.uint8))
        assert one.dtype==expected and row['output']['finite']and row['repeat_bits_equal']
        row['supported']=True
    except BaseException as error:row['supported']=False;row['failure']={'type':type(error).__name__,'message':str(error)}
    r['operators'][name]=row;writer.write(r);print(json.dumps({'operator':name,'supported':row['supported'],'failure':row.get('failure')}),flush=True)


try:
    signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in()).throw(SystemExit('CPU FP16 preflight interrupted')))
    torch.set_num_threads(8);torch.manual_seed(781234)
    r['torch_num_threads']=torch.get_num_threads();r['mkldnn_enabled']=torch.backends.mkldnn.enabled
    r['HF_modeling_source_sha256']=hashlib.sha256(Path(hf.__file__).read_bytes()).hexdigest()
    r['torch_Linear_source_sha256']=hashlib.sha256(Path(inspect.getsourcefile(nn.Linear)).read_bytes()).hexdigest()
    x=torch.randn((33,5120),dtype=torch.float16);w=torch.randn((1024,5120),dtype=torch.float16)*.01
    probe('linear_FP16_inputs_no_adapter',F.linear,(x,w))
    q=torch.randn((1,8,33,128),dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
    probe('SDPA_FP16_causal',lambda q,k,v:F.scaled_dot_product_attention(q,k,v,is_causal=True),(q,k,v))
    probe('SDPA_FP16_vision_noncausal',lambda q,k,v:F.scaled_dot_product_attention(q,k,v,is_causal=False),(q,k,v))
    norm_x=torch.randn((33,1152),dtype=torch.float16);norm_w=torch.ones(1152,dtype=torch.float16);norm_b=torch.zeros_like(norm_w)
    probe('LayerNorm_FP16_inputs',lambda x,w,b:F.layer_norm(x,(1152,),w,b,1e-6),(norm_x,norm_w,norm_b))
    rms=hf.Qwen3VLTextRMSNorm(5120,eps=1e-6).to(dtype=torch.float16)
    probe('original_HF_RMSNorm_FP16_with_declared_FP32_reduction',rms,(x,))
    conv_x=torch.randn((2,3,2,16,16),dtype=torch.float16);conv_w=torch.randn((16,3,2,16,16),dtype=torch.float16)*.01
    probe('Conv3d_FP16_patch_embedding',lambda x,w:F.conv3d(x,w,stride=(2,16,16)),(conv_x,conv_w))
    pixels=torch.randn((1,4,12,12),dtype=torch.float16)
    probe('bilinear_interpolation_FP16',lambda x:F.interpolate(x,size=(18,24),mode='bilinear',align_corners=False),(pixels,))
    weight=torch.randn((1,4,12,12),dtype=torch.float16)
    probe('bicubic_interpolation_FP16',lambda x:F.interpolate(x,size=(18,24),mode='bicubic',align_corners=False),(weight,))
    gate=torch.randn((33,1024),dtype=torch.float16);up=torch.randn_like(gate)
    probe('SwiGLU_FP16_original_pointwise',lambda a,b:F.silu(a)*b,(gate,up))
    r['all_required_preflight_operators_supported']=all(row['supported']for row in r['operators'].values())
    r['operator_coverage_does_not_prove_whole_model_accuracy']=True;r['completed']=True
    assert r['all_required_preflight_operators_supported']
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
