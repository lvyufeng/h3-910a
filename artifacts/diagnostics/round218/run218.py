#!/usr/bin/env python3
"""Run current same-process clean/profile/restored DiT diagnostic, no candidate."""
from pathlib import Path
import os
import sys
root=Path('/tmp/h3_path218')
rank=int(os.environ['LOCAL_RANK'])
import torch
import torch_npu
# Apply only private kernel-cache location, not the installed/shared wrapper.
from vllm_omni.diffusion.attention.backends import fa910a
fa910a._KERNEL_DIR=str(root/'fa_kernels')
torch.npu.set_device(rank)
sys.argv=['benchmark_path218.py',
          '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
          '--mode','baseline','--label','path218','--tp','2','--usp','4',
          '--offload','native','--audit','strict','--audit-sequence','strict,profile,strict',
          '--steps','25','--warmup','1','--repeats','2']
import benchmark_path218 as benchmark
benchmark.ROOT=root
benchmark.main()
