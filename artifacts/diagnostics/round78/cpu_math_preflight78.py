#!/usr/bin/env python3
"""Explicit CPU math-SDPA FP16 interface support,independent from model-reference execution."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from evidence_io78 import EvidenceWriter

ROOT=Path('/tmp/h3_round78_preflight01')
writer=EvidenceWriter(ROOT/'math_preflight.json')
r={'scope':__doc__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch':torch.__version__,
   'backend':'SDPBackend.MATH','FP32_internal_accumulation_is_standard_PyTorch':True,'BF16_not_used':True,'cases':[],'CPU_model_reference_generated':False}
try:
    torch.set_num_threads(4);torch.manual_seed(78)
    for causal,dim in((True,128),(False,72)):
        q=torch.randn((1,8,33,dim),dtype=torch.float16);k=torch.randn_like(q);v=torch.randn_like(q)
        with torch.inference_mode(),sdpa_kernel(backends=[SDPBackend.MATH]):
            a=F.scaled_dot_product_attention(q,k,v,is_causal=causal);b=F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        assert a.dtype==torch.float16 and a.device.type=='cpu'and torch.isfinite(a).all()
        assert torch.equal(a.view(torch.uint8),b.view(torch.uint8))
        r['cases'].append({'causal':causal,'head_dim':dim,'dtype':str(a.dtype),'finite':True,'repeat_bits_equal':True,
                           'sha256':hashlib.sha256(memoryview(a.contiguous().view(torch.uint8).numpy())).hexdigest()})
    r['supported']=True;r['completed']=True
except BaseException as error:r['failure']={'type':type(error).__name__,'message':str(error)};raise
finally:writer.write(r)
print(json.dumps(r,indent=2))
