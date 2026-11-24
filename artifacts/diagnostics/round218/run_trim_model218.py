#!/usr/bin/env python3
"""8-rank one-candidate measurement, unchanged TP2+USP4 and 155 strict checks."""
import os
from pathlib import Path
import sys
import torch
import torch_npu
root=Path('/tmp/h3_trim_model218')
torch.npu.set_device(int(os.environ['LOCAL_RANK']))
from vllm_omni.diffusion.attention.backends import fa910a
fa910a._KERNEL_DIR=str(root/'baseline_fa_kernels')
sys.argv=['benchmark_trim_model218.py',
 '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
 '--mode','baseline','--label','single_trim218','--tp','2','--usp','4',
 '--offload','native','--audit','strict','--audit-sequence','strict,trim,strict',
 '--steps','25','--warmup','1','--repeats','2']
import benchmark_trim_model218 as bench
bench.ROOT=root
bench.main()
