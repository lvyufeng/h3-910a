"""Self-developed Ascend910A TIK vector kernels with explicit rounding boundaries.

Kernels accept FP32 GM inputs. BF16 conversion at API boundaries is handled by
Torch, but every *intermediate* BF16 round is software RNE in TIK; 910A has no
native BF16 vector cast. No global precision flags or production source changes.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/tmp/h3_round58_vec01')
KERNELS = ROOT / 'kernels'
_LIB = None
_HANDLES = {}


def library():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL('/tmp/libfa910a.so')
        lib.fa910a_init.argtypes = [ctypes.c_int]
        lib.fa910a_init.restype = ctypes.c_int
        lib.fa910a_register.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.fa910a_register.restype = ctypes.c_void_p
        lib.fa910a_launch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_void_p] * 6
        lib.fa910a_launch.restype = ctypes.c_int
        assert lib.fa910a_init(0) == 0
        _LIB = lib
    return _LIB


def core_budget():
    import torch
    lib = library()
    lib.aclrtGetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    lib.aclrtGetDeviceInfo.restype = ctypes.c_int
    value = ctypes.c_int64()
    assert lib.aclrtGetDeviceInfo(torch.npu.current_device(), 101, ctypes.byref(value)) == 0
    assert 0 < value.value <= 64
    return int(value.value)


def _round_bf16(t, src, dst, parity, parity_float, bias, parity_mask, clear_mask, width):
    """Bit-exact FP32->BF16->FP32 round-to-nearest-even for finite values.

    910A has int32 add/conversion and uint16 AND, but no vector shifts. Extract
    bit16 with uint16 AND, interpret as int32 0/65536, convert/scale exactly to
    int32 0/1, then add 0x7fff+LSB and clear the low halfword.
    """
    repeats = width // 64
    t.vand(128, parity.reinterpret_cast_to('uint16'), src.reinterpret_cast_to('uint16'),
           parity_mask.reinterpret_cast_to('uint16'), repeats, 1, 1, 1, 8, 8, 0)
    t.vconv(64, 'none', parity_float, parity, repeats, 1, 1, 8, 8)
    t.vmuls(64, parity_float, parity_float, 1.0 / 65536.0, repeats, 1, 1, 8, 8)
    t.vconv(64, 'round', parity, parity_float, repeats, 1, 1, 8, 8)
    t.vadd(64, dst.reinterpret_cast_to('int32'), src.reinterpret_cast_to('int32'), bias,
           repeats, 1, 1, 1, 8, 8, 0)
    t.vadd(64, dst.reinterpret_cast_to('int32'), dst.reinterpret_cast_to('int32'), parity,
           repeats, 1, 1, 1, 8, 8, 8)
    t.vand(128, dst.reinterpret_cast_to('uint16'), dst.reinterpret_cast_to('uint16'),
           clear_mask.reinterpret_cast_to('uint16'), repeats, 1, 1, 1, 8, 8, 0)


def _round_fp16(t, src, dst, half, width):
    t.vconv(64, 'none', half, src, width // 64, 1, 1, 4, 8)
    t.vconv(64, 'none', dst, half, width // 64, 1, 1, 8, 4)


def compile_kernel(kind, rows, width, dtype='bf16', eps=1e-6, qwen=True):
    assert kind in ('round', 'rms', 'addrms', 'swiglu')
    assert width % 64 == 0 and 64 <= width <= 8192 and rows > 0
    assert dtype in ('bf16', 'fp16', 'fp32')
    cores = min(core_budget(), rows)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    identity = f'vec58_{kind}_r{rows}_d{width}_{dtype}_q{int(qwen)}_c{cores}_{source_hash}'
    identity += '_e' + hashlib.sha256(str(eps).encode()).hexdigest()[:8]
    target = KERNELS / identity
    obj = target / (identity + '.o')
    if obj.exists():
        meta = json.loads(obj.with_suffix('.json').read_text())
        assert meta['blockDim'] == cores
        return str(obj), identity + '__kernel0', cores
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    t = tik.Tik(disable_debug=True)
    x_gm = t.Tensor('float32', (rows, width), name='x_gm', scope=tik.scope_gm)
    # RMS uses a single gamma row; SwiGLU uses a complete up tensor.
    g_gm = t.Tensor('float32', (rows, width) if kind == 'swiglu' else (width,), name='g_gm', scope=tik.scope_gm)
    r_gm = t.Tensor('float32', (rows, width), name='r_gm', scope=tik.scope_gm)
    y_gm = t.Tensor('float32', (rows, width), name='y_gm', scope=tik.scope_gm)
    merged_gm = t.Tensor('float32', (rows, width), name='merged_gm', scope=tik.scope_gm)
    with t.for_range(0, cores, block_num=cores) as core:
        x = t.Tensor('float32', (width,), name='x', scope=tik.scope_ubuf)
        gamma = t.Tensor('float32', (width,), name='gamma', scope=tik.scope_ubuf)
        tmp = t.Tensor('float32', (width,), name='tmp', scope=tik.scope_ubuf)
        dst = t.Tensor('float32', (width,), name='dst', scope=tik.scope_ubuf)
        parity = t.Tensor('int32', (width,), name='parity', scope=tik.scope_ubuf)
        parity_float = t.Tensor('float32', (width,), name='parity_float', scope=tik.scope_ubuf)
        bias = t.Tensor('int32', (64,), name='bias', scope=tik.scope_ubuf)
        parity_mask = t.Tensor('int32', (64,), name='parity_mask', scope=tik.scope_ubuf)
        clear_mask = t.Tensor('int32', (64,), name='clear_mask', scope=tik.scope_ubuf)
        t.vector_dup(64, bias, 32767, 1, 1, 8)
        t.vector_dup(64, parity_mask, 65536, 1, 1, 8)
        t.vector_dup(64, clear_mask, -65536, 1, 1, 8)
        half = t.Tensor('float16', (width,), name='half', scope=tik.scope_ubuf)
        if kind in ('rms', 'addrms'):
            sum_dst = t.Tensor('float32', (8,), name='sum_dst', scope=tik.scope_ubuf)
            sum_work = t.Tensor('float32', (((width // 64 + 7) // 8) * 8,), name='sum_work', scope=tik.scope_ubuf)
            variance = t.Tensor('float32', (64,), name='variance', scope=tik.scope_ubuf)
            inv = t.Tensor('float32', (64,), name='inv', scope=tik.scope_ubuf)
            correction = t.Tensor('float32', (64,), name='correction', scope=tik.scope_ubuf)
            t.data_move(gamma, g_gm, 0, 1, width // 8, 0, 0)
        with t.for_range(0, (rows + cores - 1) // cores) as step:
            row = core + step * cores
            with t.if_scope(row < rows):
                t.data_move(x, x_gm[row, 0], 0, 1, width // 8, 0, 0)
                if kind == 'addrms':
                    t.data_move(tmp, r_gm[row, 0], 0, 1, width // 8, 0, 0)
                    t.h_add(x, x, tmp)
                    if dtype == 'bf16':
                        _round_bf16(t, x, x, parity, parity_float, bias, parity_mask, clear_mask, width)
                    elif dtype == 'fp16':
                        _round_fp16(t, x, x, half, width)
                if kind == 'round':
                    _round_bf16(t, x, dst, parity, parity_float, bias, parity_mask, clear_mask, width)
                elif kind in ('rms', 'addrms'):
                    t.h_mul(tmp, x, x)
                    t.vec_reduce_add(64, sum_dst, tmp, sum_work, width // 64, 8)
                    scalar_variance = t.Scalar('float32', name='scalar_variance')
                    scalar_variance.set_as(sum_dst[0] * (1.0 / width) + eps)
                    t.vector_dup(64, variance, scalar_variance, 1, 1, 8)
                    t.vec_rsqrt(64, inv, variance, 1, 8, 8)
                    # Two Newton steps for FP32 rsqrt precision, no fusion of
                    # multiply/add that changes the stated FP32 reference order.
                    for _ in range(2):
                        t.h_mul(correction, inv, inv)
                        t.h_mul(correction, correction, variance)
                        t.h_mul(correction, correction, -0.5)
                        t.h_add(correction, correction, 1.5)
                        t.h_mul(inv, inv, correction)
                    scale = t.Scalar('float32', name='scale')
                    scale.set_as(inv[0])
                    t.vmuls(64, dst, x, scale, width // 64, 1, 1, 8, 8)
                    if dtype == 'bf16' and qwen:
                        _round_bf16(t, dst, dst, parity, parity_float, bias, parity_mask, clear_mask, width)
                    t.h_mul(dst, dst, gamma)
                else:
                    t.data_move(gamma, g_gm[row, 0], 0, 1, width // 8, 0, 0)
                    # Stable sigmoid: exp(-abs(x)) <= 1, avoiding exp overflow
                    # for finite negative outliers and preserving signed zero.
                    t.h_abs(tmp, x)
                    t.h_mul(tmp, tmp, -1.0)
                    t.h_exp(tmp, tmp)
                    t.h_add(dst, tmp, 1.0)
                    t.h_rec(parity_float, dst)
                    for _ in range(2):
                        t.h_mul(parity.reinterpret_cast_to('float32'), dst, parity_float)
                        t.h_mul(parity.reinterpret_cast_to('float32'), parity.reinterpret_cast_to('float32'), -1.0)
                        t.h_add(parity.reinterpret_cast_to('float32'), parity.reinterpret_cast_to('float32'), 2.0)
                        t.h_mul(parity_float, parity_float, parity.reinterpret_cast_to('float32'))
                    t.h_mul(tmp, tmp, parity_float)
                    zero = t.Tensor('float32', (64,), name='zero', scope=tik.scope_ubuf)
                    t.vector_dup(64, zero, 0.0, 1, 1, 8)
                    for i in range(width // 64):
                        mask = t.vcmp_ge(64, x[i * 64], zero, 1, 1)
                        t.vsel(64, 0, dst[i * 64], mask, parity_float[i * 64], tmp[i * 64], 1, 1, 1, 1, 8, 8, 8)
                    t.h_mul(dst, dst, x)
                    if dtype == 'bf16':
                        _round_bf16(t, dst, dst, parity, parity_float, bias, parity_mask, clear_mask, width)
                    elif dtype == 'fp16':
                        _round_fp16(t, dst, dst, half, width)
                    t.h_mul(dst, dst, gamma)
                t.data_move(y_gm[row, 0], dst, 0, 1, width // 8, 0, 0)
                # The fixed five-pointer existing launch ABI includes an updated
                # residual output even for plain norm/activation. Wrappers do not
                # claim to fuse external BF16 input/output casts into this kernel.
                t.data_move(merged_gm[row, 0], x, 0, 1, width // 8, 0, 0)
    KERNELS.mkdir(exist_ok=True)
    work = os.path.realpath(tempfile.mkdtemp(prefix='vec58_build_', dir=ROOT))
    if os.path.dirname(work) != str(ROOT.resolve()) or not os.path.basename(work).startswith('vec58_build_'):
        raise RuntimeError('Unexpected build path')
    old = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=identity, inputs=[x_gm, g_gm, r_gm], outputs=[y_gm, merged_gm])
        source = Path(work) / 'kernel_meta' / (identity + '.o')
        assert source.exists() and source.resolve() == source.absolute()
        target.mkdir(exist_ok=False)
        shutil.copyfile(source.with_suffix('.json'), obj.with_suffix('.json'))
        shutil.copyfile(source, obj)
    finally:
        os.chdir(old)
        if Path(work).resolve() == Path(work).absolute() and Path(work).parent == ROOT.resolve():
            shutil.rmtree(work)
    meta = json.loads(obj.with_suffix('.json').read_text())
    assert meta['blockDim'] == cores
    return str(obj), identity + '__kernel0', cores


def launch(kind, x, g, residual=None, dtype='bf16', eps=1e-6, qwen=True):
    import torch
    assert x.dtype == g.dtype == torch.float32 and x.device.type in ('npu', 'privateuseone')
    assert x.is_contiguous() and g.is_contiguous() and x.device == g.device
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    obj, symbol, cores = compile_kernel(kind, rows, width, dtype, eps, qwen)
    if obj not in _HANDLES:
        handle = library().fa910a_register(obj.encode(), symbol.encode())
        if not handle:
            raise RuntimeError('Vector kernel registration failed')
        _HANDLES[obj] = handle
    y = torch.empty_like(x)
    updated = torch.empty_like(x)
    residual = x if residual is None else residual
    assert residual.dtype == x.dtype and residual.device == x.device and residual.is_contiguous()
    stream = torch.npu.current_stream().npu_stream
    rc = library().fa910a_launch(_HANDLES[obj], symbol.encode(), cores,
                                 x.data_ptr(), g.data_ptr(), residual.data_ptr(), y.data_ptr(), updated.data_ptr(), ctypes.c_void_p(stream))
    if rc:
        raise RuntimeError('Vector kernel launch failed rc=%s' % rc)
    return y, updated
