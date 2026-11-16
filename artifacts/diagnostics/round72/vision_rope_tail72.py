"""910A vision RoPE partial-tail fusion; unsupported gather prototype is separate.

Original Torch pitched-BF16->FP32 cast and rotate_half remain unchanged. This
kernel only fuses two separate FP32 multiplies, add and one BF16 RNE/pack.
No unaligned half36 DMA, gather, FMA, clamp or IEEE/signed-zero correction.
Finite fixture certification is required; this is diagnostic, not production.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import torch
import torch_npu
sys.path.insert(0, '/tmp/h3_round58_vec01')
from native_bf16_vector58 import pack_bf16
from fused_vector58 import library, core_budget, _round_bf16

ROOT = Path('/tmp/h3_round72_tail01')
KERNELS = ROOT / 'kernels'
_HASH = hashlib.sha256(Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()
                       + Path('/tmp/h3_round58_vec01/fused_vector58.py').read_bytes()).hexdigest()[:12]
_CACHE, _HANDLES = {}, {}


def build(tokens, cores):
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    width, d, heads, padded = 1152, 72, 16, 1280
    name = f'vision_rope_tail72_n{tokens}_d72_h16_c{cores}_{_HASH}'
    obj = KERNELS / name / (name + '.o')
    if obj.is_file():
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores
    t = tik.Tik(disable_debug=True)
    xgm = t.Tensor('float32', (tokens, width), name='x_gm', scope=tik.scope_gm)
    rgm = t.Tensor('float32', (tokens, width), name='rot_gm', scope=tik.scope_gm)
    cgm = t.Tensor('float32', (tokens, d), name='cos_gm', scope=tik.scope_gm)
    sgm = t.Tensor('float32', (tokens, d), name='sin_gm', scope=tik.scope_gm)
    ygm = t.Tensor('uint16', (tokens, width), name='output_gm', scope=tik.scope_gm)
    with t.for_range(0, cores, block_num=cores) as core:
        x = t.Tensor('float32', (padded,), name='x', scope=tik.scope_ubuf)
        rotated = t.Tensor('float32', (padded,), name='rotated', scope=tik.scope_ubuf)
        cos = t.Tensor('float32', (padded,), name='cos', scope=tik.scope_ubuf)
        sin = t.Tensor('float32', (padded,), name='sin', scope=tik.scope_ubuf)
        out = t.Tensor('float32', (padded,), name='out', scope=tik.scope_ubuf)
        words = t.Tensor('uint16', (padded,), name='words', scope=tik.scope_ubuf)
        transposed = t.Tensor('uint16', (padded,), name='transposed', scope=tik.scope_ubuf)
        discarded = t.Tensor('uint16', (padded,), name='discarded', scope=tik.scope_ubuf)
        parity = t.Tensor('int32', (padded,), name='parity', scope=tik.scope_ubuf)
        parity_float = t.Tensor('float32', (padded,), name='parity_float', scope=tik.scope_ubuf)
        bias = t.Tensor('int32', (64,), name='bias', scope=tik.scope_ubuf)
        parity_mask = t.Tensor('int32', (64,), name='parity_mask', scope=tik.scope_ubuf)
        clear_mask = t.Tensor('int32', (64,), name='clear_mask', scope=tik.scope_ubuf)
        t.vector_dup(64, bias, 32767, 1, 1, 8)
        t.vector_dup(64, parity_mask, 65536, 1, 1, 8)
        t.vector_dup(64, clear_mask, -65536, 1, 1, 8)
        with t.for_range(0, (tokens + cores - 1) // cores) as step:
            row = core + step * cores
            with t.if_scope(row < tokens):
                for tensor in (x, rotated, cos, sin):
                    t.vector_dup(64, tensor, 0.0, padded // 64, 1, 8)
                t.data_move(x, xgm[row, 0], 0, 1, width // 8, 0, 0)
                t.data_move(rotated, rgm[row, 0], 0, 1, width // 8, 0, 0)
                t.data_move(cos, cgm[row, 0], 0, 1, d // 8, 0, 0)
                t.data_move(sin, sgm[row, 0], 0, 1, d // 8, 0, 0)
                for head in range(1, heads):
                    t.data_move(cos[head * d], cos, 0, 1, d // 8, 0, 0)
                    t.data_move(sin[head * d], sin, 0, 1, d // 8, 0, 0)
                t.h_mul(out, x, cos)
                t.h_mul(rotated, rotated, sin)
                t.h_add(out, out, rotated)
                _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                pack_bf16(t, out, words, transposed, discarded, padded)
                t.data_move(ygm[row, 0], words, 0, 1, width // 16, 0, 0)
    work = Path(tempfile.mkdtemp(prefix='vision_tail72_build_', dir=ROOT)).resolve(strict=True)
    if work.parent != ROOT.resolve() or not work.name.startswith('vision_tail72_build_'):
        raise RuntimeError('Unexpected tail build directory')
    old = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name, inputs=[xgm, rgm, cgm, sgm], outputs=[ygm])
        src = work / 'kernel_meta' / (name + '.o')
        assert src.is_file()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'), obj.with_suffix('.json'))
        shutil.copyfile(src, obj)
    finally:
        os.chdir(old)
        if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('vision_tail72_build_'):
            raise RuntimeError('Unexpected tail cleanup target')
        shutil.rmtree(work)
    assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
    return str(obj), name + '__kernel0', cores


def tail(x, rotated, cos, sin):
    if x.ndim != 3 or tuple(x.shape[1:]) != (16, 72) or rotated.shape != x.shape or not 0 < x.shape[0] <= 16384:
        raise ValueError('Measured vision RoPE tail requires[N,16,72]')
    n = x.shape[0]
    if (any(value.dtype != torch.float32 or value.device != x.device or not value.is_contiguous()
            or value.data_ptr() % 32 or torch_npu.get_npu_format(value) not in (0, 2) for value in (x, rotated, cos, sin))
            or x.device.type not in ('npu', 'privateuseone') or x.device.index != torch.npu.current_device()
            or cos.shape != sin.shape or tuple(cos.shape) != (n, 72)):
        raise ValueError('Vision RoPE tail requires current-device aligned denseFP32 operands and coefficients')
    if torch.is_grad_enabled() and any(value.requires_grad for value in (x, rotated, cos, sin)):
        raise ValueError('Inference-only vision RoPE tail')
    cores = min(n, core_budget())
    key = (n, cores)
    if key not in _CACHE:
        KERNELS.mkdir(exist_ok=True)
        with (KERNELS / '.compile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _CACHE[key] = build(n, cores)
    obj, symbol, cores = _CACHE[key]
    if obj not in _HANDLES:
        handle = library().fa910a_register(obj.encode(), symbol.encode())
        if not handle:
            raise RuntimeError('Vision tail registration failed')
        _HANDLES[obj] = handle
    output = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    if output.data_ptr() % 32 or not output.is_contiguous() or torch_npu.get_npu_format(output) not in (0, 2):
        raise RuntimeError('Invalid dense vision tail output')
    rc = library().fa910a_launch(_HANDLES[obj], symbol.encode(), cores, x.data_ptr(), rotated.data_ptr(), cos.data_ptr(),
                                 sin.data_ptr(), output.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Vision RoPE tail launch failed:%d' % rc)
    return output


def rope_pair(q, k, cos, sin):
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or q.shape != k.shape or tuple(q.shape[1:]) != (16, 72):
        raise ValueError('Measured vision wrapper requires BF16 sibling[N,16,72] inputs')
    # Deliberately keep the original Torch cast and cat/negate rotation. This is
    # partial-tail fusion, not an unproven replacement of packed input rotation.
    q32, k32 = q.float(), k.float()
    rq = torch.cat((-q32[..., 36:], q32[..., :36]), dim=-1)
    rk = torch.cat((-k32[..., 36:], k32[..., :36]), dim=-1)
    return tail(q32, rq, cos, sin), tail(k32, rk, cos, sin)
