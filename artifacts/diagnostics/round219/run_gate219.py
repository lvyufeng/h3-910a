#!/usr/bin/env python3
"""R219 supplemental gate-only replay; every model output comes from original FA."""
import os
from pathlib import Path
import sys
import torch
import torch_npu
root=Path('/tmp/h3_gate219')
torch.npu.set_device(int(os.environ['LOCAL_RANK']))
from vllm_omni.diffusion.attention.backends import fa910a
fa910a._KERNEL_DIR=str(root/'baseline_fa_kernels')
sys.argv=['benchmark_gate219.py',
 '--checkpoint','/home/mseco/models/MiniMax-H3/FL2VA/transformer',
 '--mode','baseline','--label','gate219','--tp','2','--usp','4',
 '--offload','native','--audit','strict','--audit-sequence','strict,gate,strict',
 '--steps','0,48','--warmup','1','--repeats','1']
import benchmark_gate219 as bench
bench.ROOT=root
bench.main()
