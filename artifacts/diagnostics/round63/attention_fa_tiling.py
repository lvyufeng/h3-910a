#!/usr/bin/env python3
"""Exploratory short-D64 FA tiling on actual VAE QKV; never a lossless pass.

Reuse the deployed TBE builder and current-stream launcher, but compile wider
KV tiles under distinct br/bc cache identities. Raw-NZ timing excludes packing;
actual output checks include the same complete head sequence and exact key mask.
Run only after the eight-card decoder job has finished.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch_npu
from vllm_omni.platforms import current_omni_platform
from vllm_omni.diffusion.attention.backends import fa910a as fa
from evidence_io63 import EvidenceWriter

ROOT = Path('/tmp/h3_round63_decode01')
CAP = ROOT / 'capture' / 'rank0'
OUT = ROOT / 'attention_fa_tiling01.json'
writer = EvidenceWriter(OUT)
current_omni_platform.set_device(torch.device('npu', 0))
torch.set_num_threads(8)
r = {'cases': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'evidence_writer_sha256': hashlib.sha256((ROOT / 'evidence_io63.py').read_bytes()).hexdigest(),
     'scope': 'Exploratory actual VAE D64 attention TBE tiling; raw-NZ timings exclude packing/allocation. Different from legacy FP16 softmax arithmetic. Not accepted lossless or whole-video.'}


def metric(a, b):
    x, y = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    e = x.double() - y.double()
    return {'raw_bits_equal': torch.equal(x.view(torch.uint8), y.view(torch.uint8)),
            'elements': x.numel(), 'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'word_mismatches': int(x.view(torch.int16).ne(y.view(torch.int16)).sum()),
            'max_abs_error': float(e.abs().max()),
            'relative_rms': float(e.square().mean().sqrt() / y.double().square().mean().sqrt().clamp_min(1e-30))}


try:
    with torch.inference_mode():
        data = torch.load(CAP / 'attention0_inputs.pt', weights_only=True)
        n, padded, d, heads = 1797, 2048, 64, 32
        nz_shape = (1, heads, d // 16, padded // 16, 16, 16)
        nz = []
        for key in ('q', 'k', 'v'):
            x = torch.nn.functional.pad(data[key].to('npu').transpose(1, 2).contiguous(), (0, 0, 0, padded - n))
            if key == 'q':
                x = x * (d ** -.5)
            nz.append(x.reshape(1, heads, padded // 16, 16, d // 16, 16).permute(0, 1, 4, 2, 3, 5).contiguous())
        out = torch.empty(nz_shape, dtype=torch.float16, device='npu')
        lse = torch.empty((1, heads, padded), dtype=torch.float32, device='npu')
        actual = torch.load(CAP / 'attention0_output.pt', weights_only=True)
        stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        lib = fa._lib()
        for br, bc in ((128, 256), (128, 512), (64, 1024), (32, 2048), (16, 2048)):
            row = {'br': br, 'bc': bc}
            t = time.perf_counter()
            try:
                obj, symbol, blocks = fa._build(1, heads, padded, d, br_cap=br, bc_cap=bc, key_seq_limit=n)
            except Exception as exc:
                # Build-only failure: nothing was registered/launched for this case.
                row['build_failure'] = {'type': type(exc).__name__, 'message': str(exc)}
                r['cases'].append(row)
                writer.write(r)
                print(json.dumps(row), flush=True)
                continue
            row.update(build_or_cache_s=time.perf_counter() - t, object=obj, symbol=symbol, block_dim=blocks)
            handle = lib.fa910a_register(obj.encode(), symbol.encode())
            if not handle:
                raise RuntimeError('FA registration failed')
            def launch():
                rc = lib.fa910a_launch(handle, symbol.encode(), blocks, *[x.data_ptr() for x in nz], out.data_ptr(), lse.data_ptr(), stream)
                if rc:
                    raise RuntimeError('FA launch failed: %d' % rc)
            launch()
            torch.npu.synchronize()
            first = out.permute(0, 1, 3, 4, 2, 5).reshape(1, heads, padded, d)[..., :n, :].transpose(1, 2).contiguous().cpu()
            groups = []
            for _ in range(3):
                torch.npu.synchronize()
                t = time.perf_counter()
                for _ in range(5):
                    launch()
                torch.npu.synchronize()
                groups.append((time.perf_counter() - t) * 1000 / 5)
            value = out.permute(0, 1, 3, 4, 2, 5).reshape(1, heads, padded, d)[..., :n, :].transpose(1, 2).contiguous().cpu()
            row.update(raw_groups_ms=groups, median_raw_ms=statistics.median(groups),
                       first_vs_repeat=metric(value, first), vs_actual=metric(value, actual))
            r['cases'].append(row)
            writer.write(r)
            print(json.dumps(row), flush=True)
        r['completed'] = True
except BaseException as exc:
    r['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
    raise
finally:
    writer.write(r)
    print('Evidence:', OUT, flush=True)
