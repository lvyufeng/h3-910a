#!/usr/bin/env python3
"""Prebuild two verified row-GEMM binaries once before eight-rank model loading.

Source-only setup. This script compiles but launches no kernel and changes no
model weights or production code. Each model rank still verifies the objects.
"""
import hashlib
import json
from pathlib import Path
import blocked_dynamic_steps83 as kernel

ROOT=Path('/tmp/h3_round84_row_fp3202')
PIN='b5a06cfdb4c83a2d2f8c84eb56e8794f8a38d2732c4ff5e1fbef0901b520facb'
if hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest()!=PIN:
    raise RuntimeError('Executed R83 FP32 GEMM source changed')
receipt=ROOT/'prebuilt_row_kernels84.json'
if receipt.exists():raise RuntimeError('Refusing to overwrite row-kernel build receipt')
records=[kernel.build_kernel(m=2043,k=k,n=5120,out_dir=ROOT/'kernels')for k in(1024,3200)]
assert all(item['dst_dtype']=='float32'for item in records)
with receipt.open('x')as f:f.write(json.dumps({'scope':__doc__,'kernel_source_sha256':PIN,'objects':records,
                                            'kernel_launch_executed':False},indent=2)+'\n')
print(json.dumps({'prebuilt_shapes':[item['logical_shape']for item in records],'no_kernel_launch':True},indent=2))
