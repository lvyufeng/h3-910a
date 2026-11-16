#!/usr/bin/env python3
"""Read raw TIK square/sum/mean/rsqrt stages for finite-overflow counterexamples."""
import ctypes
import json
import os
from pathlib import Path
import tempfile
import torch
import torch_npu
from tbe import tik
import tbe.common.platform as platform
from fused_vector58 import ROOT, library

OUT = ROOT / 'overflow_stages.json'
if OUT.exists():
    raise RuntimeError('Refuse overflow stage evidence overwrite')
torch.npu.set_device(0)
platform.set_current_compile_soc_info('Ascend910A')
t = tik.Tik(disable_debug=True)
rows, width = 8, 128
xgm = t.Tensor('float32', (rows, width), name='stage_xgm', scope=tik.scope_gm)
ggm = t.Tensor('float32', (width,), name='stage_ggm', scope=tik.scope_gm)
rgm = t.Tensor('float32', (rows, width), name='stage_rgm', scope=tik.scope_gm)
ygm = t.Tensor('float32', (rows, width), name='stage_ygm', scope=tik.scope_gm)
ogm = t.Tensor('float32', (rows, width), name='stage_ogm', scope=tik.scope_gm)
with t.for_range(0, rows, block_num=rows) as row:
    x = t.Tensor('float32', (width,), name='stage_x', scope=tik.scope_ubuf)
    sq = t.Tensor('float32', (width,), name='stage_sq', scope=tik.scope_ubuf)
    total = t.Tensor('float32', (8,), name='stage_total', scope=tik.scope_ubuf)
    work = t.Tensor('float32', (8,), name='stage_work', scope=tik.scope_ubuf)
    mean = t.Tensor('float32', (8,), name='stage_mean', scope=tik.scope_ubuf)
    inv = t.Tensor('float32', (8,), name='stage_inv', scope=tik.scope_ubuf)
    stats = t.Tensor('float32', (width,), name='stage_stats', scope=tik.scope_ubuf)
    t.data_move(x, xgm[row, 0], 0, 1, width // 8, 0, 0)
    t.h_mul(sq, x, x)
    t.vec_reduce_add(64, total, sq, work, width // 64, 8)
    t.vmuls(8, mean, total, 1.0 / width, 1, 1, 1, 1, 1)
    t.vadds(8, mean, mean, 1e-6, 1, 1, 1, 1, 1)
    t.vec_rsqrt(8, inv, mean, 1, 1, 1)
    t.vector_dup(64, stats, 0., width // 64, 1, 8)
    stats[0].set_as(total[0]); stats[1].set_as(mean[0]); stats[2].set_as(inv[0])
    t.data_move(ygm[row, 0], sq, 0, 1, width // 8, 0, 0)
    t.data_move(ogm[row, 0], stats, 0, 1, width // 8, 0, 0)
name = 'vector58_overflow_stages'
with tempfile.TemporaryDirectory(prefix='overflow58_', dir=ROOT) as workdir:
    assert Path(workdir).resolve().parent == ROOT.resolve() and Path(workdir).name.startswith('overflow58_')
    old = os.getcwd()
    try:
        os.chdir(workdir)
        t.BuildCCE(kernel_name=name, inputs=[xgm, ggm, rgm], outputs=[ygm, ogm])
        obj = str(Path(workdir) / 'kernel_meta' / (name + '.o'))
        lib = library(); h = lib.fa910a_register(obj.encode(), (name + '__kernel0').encode()); assert h
        vals = torch.tensor([0., 1., 1e15, 1e18, 1e19, 1e20, -1e20, 2.0**64], dtype=torch.float32)
        # BF16 input values are decoded exactly before the same FP32 math.
        cpu = vals.bfloat16().float().reshape(rows, 1).expand(rows, width).contiguous()
        x = cpu.to('npu'); y = torch.empty_like(x); stats = torch.empty_like(x); g = torch.ones(width, device='npu')
        rc = lib.fa910a_launch(h, (name + '__kernel0').encode(), rows, x.data_ptr(), g.data_ptr(), x.data_ptr(),
                              y.data_ptr(), stats.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream)); assert rc == 0
        torch.npu.synchronize(); a, s = y.cpu(), stats.cpu()
        result = {'scope': 'Raw FP32 vector stages after exact BF16 decoding; NaN/inf strings preserve metadata JSON validity', 'cases': []}
        def val(v):
            f = float(v)
            return f if torch.isfinite(torch.tensor(f)) else str(f)
        for i in range(rows):
            result['cases'].append({'input': val(cpu[i, 0]), 'tik_square': val(a[i, 0]),
                                    'tik_sum': val(s[i, 0]), 'tik_mean_plus_eps': val(s[i, 1]), 'tik_rsqrt': val(s[i, 2]),
                                    'cpu_square': val(cpu[i, 0].square()), 'cpu_sum': val(cpu[i].square().sum()),
                                    'cpu_mean': val(cpu[i].square().mean()), 'cpu_rsqrt': val(torch.rsqrt(cpu[i].square().mean() + 1e-6))})
        result['completed'] = True
        OUT.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2), flush=True)
    finally:
        os.chdir(old)
