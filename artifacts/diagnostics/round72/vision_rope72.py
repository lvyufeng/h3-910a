"""Fused packed-Q/K vision RoPE: BF16 GM -> FP32 products/add -> one BF16 RNE.

Only the measured ND [N,16,72] sibling views of packed [N,3,16,72] QKV
are accepted. Full aligned1152-word rows are loaded; half36 is rearranged
by FP32 UB gather, never an unaligned GM DMA. No contiguous/cast input copy,
BF16 ISA, FMA, intermediate BF16 round, finite clamp or signed-zero repair.
Finite actual fixtures are certified separately; generic NaN/overflow is not.
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
from native_bf16_vector58 import unpack_bf16, pack_bf16
from fused_vector58 import library, core_budget, _round_bf16

ROOT = Path('/tmp/h3_round72_rope01')
KERNELS = ROOT / 'kernels'
_HASH = hashlib.sha256(Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()
                       + Path('/tmp/h3_round58_vec01/fused_vector58.py').read_bytes()).hexdigest()[:12]
_CACHE, _HANDLES = {}, {}


def build(tokens, cores):
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    d, heads, width, pitch, padded = 72, 16, 1152, 3456, 1280
    name = f'vision_rope72_n{tokens}_d72_h16_p3456_c{cores}_{_HASH}'
    obj = KERNELS / name / (name + '.o')
    if obj.is_file():
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores
    t = tik.Tik(disable_debug=True)
    xgm = t.Tensor('uint16', (tokens, pitch), name='qkv_gm', scope=tik.scope_gm)
    cgm = t.Tensor('float32', (tokens, d), name='cos_gm', scope=tik.scope_gm)
    sgm = t.Tensor('float32', (tokens, d), name='sin_gm', scope=tik.scope_gm)
    qgm = t.Tensor('uint16', (tokens, width), name='qout_gm', scope=tik.scope_gm)
    kgm = t.Tensor('uint16', (tokens, width), name='kout_gm', scope=tik.scope_gm)
    with t.for_range(0, cores, block_num=cores) as core:
        words = t.Tensor('uint16', (padded,), name='words', scope=tik.scope_ubuf)
        transposed = t.Tensor('uint16', (padded,), name='transpose', scope=tik.scope_ubuf)
        zeros = t.Tensor('uint16', (padded,), name='zeros', scope=tik.scope_ubuf)
        x = t.Tensor('float32', (padded,), name='x', scope=tik.scope_ubuf)
        cos = t.Tensor('float32', (padded,), name='cos', scope=tik.scope_ubuf)
        sin = t.Tensor('float32', (padded,), name='sin', scope=tik.scope_ubuf)
        rotated = t.Tensor('float32', (padded,), name='rotated', scope=tik.scope_ubuf)
        out = t.Tensor('float32', (padded,), name='out', scope=tik.scope_ubuf)
        offsets = t.Tensor('uint32', (padded,), name='rotate_offsets', scope=tik.scope_ubuf)
        parity = t.Tensor('int32', (padded,), name='parity', scope=tik.scope_ubuf)
        parity_float = t.Tensor('float32', (padded,), name='parity_float', scope=tik.scope_ubuf)
        bias = t.Tensor('int32', (64,), name='bias', scope=tik.scope_ubuf)
        parity_mask = t.Tensor('int32', (64,), name='parity_mask', scope=tik.scope_ubuf)
        clear_mask = t.Tensor('int32', (64,), name='clear_mask', scope=tik.scope_ubuf)
        t.vector_dup(64, bias, 32767, 1, 1, 8)
        t.vector_dup(64, parity_mask, 65536, 1, 1, 8)
        t.vector_dup(64, clear_mask, -65536, 1, 1, 8)
        t.vector_dup(128, zeros, 0, padded // 128, 1, 8)
        for i in range(padded):
            rotated_index = (i // d) * d + (i % d + d // 2) % d if i < width else i
            offsets[i].set_as(rotated_index * 4)
        with t.for_range(0, (tokens + cores - 1) // cores) as step:
            row = core + step * cores
            with t.if_scope(row < tokens):
                t.vector_dup(64, cos, 0.0, padded // 64, 1, 8)
                t.vector_dup(64, sin, 0.0, padded // 64, 1, 8)
                t.data_move(cos, cgm[row, 0], 0, 1, d // 8, 0, 0)
                t.data_move(sin, sgm[row, 0], 0, 1, d // 8, 0, 0)
                for head in range(1, heads):
                    t.data_move(cos[head * d], cos, 0, 1, d // 8, 0, 0)
                    t.data_move(sin[head * d], sin, 0, 1, d // 8, 0, 0)
                for side in range(2):
                    t.vector_dup(128, words, 0, padded // 128, 1, 8)
                    t.data_move(words, xgm[row, side * width], 0, 1, width // 16, 0, 0)
                    unpack_bf16(t, words, x, transposed, zeros, padded)
                    t.vgather(64, rotated, x, offsets, padded // 64, 8)
                    t.vmuls(36, rotated, rotated, -1.0, heads, 1, 1, d // 8, d // 8)
                    t.h_mul(out, x, cos)
                    t.h_mul(rotated, rotated, sin)
                    t.h_add(out, out, rotated)
                    _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    pack_bf16(t, out, words, transposed, zeros, padded)
                    if side == 0:
                        t.data_move(qgm[row, 0], words, 0, 1, width // 16, 0, 0)
                    else:
                        t.data_move(kgm[row, 0], words, 0, 1, width // 16, 0, 0)
    work = Path(tempfile.mkdtemp(prefix='vision_rope72_build_', dir=ROOT)).resolve(strict=True)
    if work.parent != ROOT.resolve() or not work.name.startswith('vision_rope72_build_'):
        raise RuntimeError('Unexpected vision RoPE build directory')
    old = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name, inputs=[xgm, cgm, sgm], outputs=[qgm, kgm])
        src = work / 'kernel_meta' / (name + '.o')
        assert src.is_file()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'), obj.with_suffix('.json'))
        shutil.copyfile(src, obj)
    finally:
        os.chdir(old)
        if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('vision_rope72_build_'):
            raise RuntimeError('Unexpected vision RoPE cleanup target')
        shutil.rmtree(work)
    assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
    return str(obj), name + '__kernel0', cores


def rope_pair(q, k, cos, sin):
    if q.ndim != 3 or k.shape != q.shape or tuple(q.shape[1:]) != (16, 72) or not 0 < q.shape[0] <= 16384:
        raise ValueError('Measured packed vision RoPE requires[N,16,72]')
    n, width, pitch = q.shape[0], 1152, 3456
    if (q.dtype != torch.bfloat16 or k.dtype != q.dtype or cos.dtype != torch.float32 or sin.dtype != cos.dtype
            or tuple(q.stride()) != (pitch, 72, 1) or k.stride() != q.stride()
            or cos.shape != sin.shape or tuple(cos.shape) != (n, 72) or not cos.is_contiguous() or not sin.is_contiguous()
            or q.device.type not in ('npu', 'privateuseone') or q.device.index != torch.npu.current_device()
            or any(value.device != q.device for value in (k, cos, sin))
            or q.untyped_storage()._cdata != k.untyped_storage()._cdata or k.data_ptr() != q.data_ptr() + width * 2
            or any(value.data_ptr() % 32 or torch_npu.get_npu_format(value) not in (0, 2) for value in (q, k, cos, sin))
            or q.storage_offset() * 2 + ((n - 1) * pitch + 2 * width) * 2 > q.untyped_storage().nbytes()):
        raise ValueError('Vision RoPE requires aligned ND packed sibling Q/K and actual contiguousFP32 coefficients')
    if torch.is_grad_enabled() and any(value.requires_grad for value in (q, k, cos, sin)):
        raise ValueError('Inference-only vision RoPE')
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
            raise RuntimeError('Packed vision RoPE registration failed')
        _HANDLES[obj] = handle
    yq = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    yk = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if any(value.data_ptr() % 32 or not value.is_contiguous() or torch_npu.get_npu_format(value) not in (0, 2) for value in (yq, yk)):
        raise RuntimeError('Invalid dense vision RoPE output layout')
    rc = library().fa910a_launch(_HANDLES[obj], symbol.encode(), cores, q.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                 yq.data_ptr(), yk.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Packed vision RoPE launch failed:%d' % rc)
    return yq, yk
