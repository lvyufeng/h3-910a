"""Native BF16-GM 910A fused vector prototypes using integer codec, not BF16 ISA.

Three uint16 matrix transposes interleave each BF16 word with a zero low
halfword; the inverse transposes extract high words. Logical row/channel order
is unchanged. This avoids external Torch BF16 casts while retaining FP32 math.
"""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
from fused_vector58 import ROOT, KERNELS, library, core_budget, _round_bf16, _fix_product_zero, _SOURCE_HASH

_HASH = hashlib.sha256((Path(__file__).read_text() + _SOURCE_HASH).encode()).hexdigest()[:12]
_COMPILED = {}
_HANDLES = {}


def unpack_bf16(t, words, values, transposed, zeros, padded):
    """BF16 uint16 words -> exactly corresponding FP32 bit patterns, all bits."""
    n = padded // 256
    t.vnchwconv(False, False, [transposed[i * 16] for i in range(16)],
                [words[i * 16] for i in range(16)], n, 16, 16)
    view = values.reinterpret_cast_to('uint16')
    for half in (0, 1):
        src = []
        for i in range(8):
            src.extend([zeros[0], transposed[(half * 8 + i) * 16]])
        t.vnchwconv(False, False, [view[i * 32 + half * 16] for i in range(16)], src, n, 32, 16)


def pack_bf16(t, values, words, transposed, discarded, padded):
    """Extract FP32 high halfwords; caller explicitly applies RNE before this."""
    n = padded // 256
    view = values.reinterpret_cast_to('uint16')
    for half in (0, 1):
        dst = []
        for i in range(8):
            dst.extend([discarded[(half * 8 + i) * 16], transposed[(half * 8 + i) * 16]])
        t.vnchwconv(False, False, dst, [view[i * 32 + half * 16] for i in range(16)], n, 16, 32)
    t.vnchwconv(False, False, [words[i * 16] for i in range(16)],
                [transposed[i * 16] for i in range(16)], n, 16, 16)


def _build(kind, rows, width, qwen, eps, cores):
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    padded = ((width + 255) // 256) * 256
    name = f'nv58_{kind}_r{rows}_d{width}_p{padded}_q{int(qwen)}_c{cores}_{_HASH}'
    name += '_e' + hashlib.sha256(str(eps).encode()).hexdigest()[:8]
    target = KERNELS / name
    obj = target / (name + '.o')
    if obj.exists():
        meta = json.loads(obj.with_suffix('.json').read_text())
        assert meta['blockDim'] == cores
        return str(obj), name + '__kernel0', cores
    t = tik.Tik(disable_debug=True)
    x_gm = t.Tensor('uint16', (rows, width * 2 if kind == 'swiglu' else width), name='native_x_gm', scope=tik.scope_gm)
    gamma_gm = t.Tensor('uint16', (width,), name='native_gamma_gm', scope=tik.scope_gm)
    residual_gm = t.Tensor('uint16', (rows, width), name='native_residual_gm', scope=tik.scope_gm)
    y_gm = t.Tensor('float32' if kind == 'codec' else 'uint16', (rows, width), name='native_y_gm', scope=tik.scope_gm)
    updated_gm = t.Tensor('uint16', (rows, width), name='native_updated_gm', scope=tik.scope_gm)
    with t.for_range(0, cores, block_num=cores) as core:
        words = t.Tensor('uint16', (padded,), name='words_ub', scope=tik.scope_ubuf)
        transposed = t.Tensor('uint16', (padded,), name='transposed_ub', scope=tik.scope_ubuf)
        zeros = t.Tensor('uint16', (padded,), name='zero_words_ub', scope=tik.scope_ubuf)
        x = t.Tensor('float32', (padded,), name='native_x', scope=tik.scope_ubuf)
        gamma = t.Tensor('float32', (padded,), name='native_gamma', scope=tik.scope_ubuf)
        tmp = t.Tensor('float32', (padded,), name='native_tmp', scope=tik.scope_ubuf)
        dst = t.Tensor('float32', (padded,), name='native_dst', scope=tik.scope_ubuf)
        parity = t.Tensor('int32', (padded,), name='native_parity', scope=tik.scope_ubuf)
        parity_float = t.Tensor('float32', (padded,), name='native_parity_float', scope=tik.scope_ubuf)
        bias = t.Tensor('int32', (64,), name='native_bias', scope=tik.scope_ubuf)
        parity_mask = t.Tensor('int32', (64,), name='native_parity_mask', scope=tik.scope_ubuf)
        clear_mask = t.Tensor('int32', (64,), name='native_clear_mask', scope=tik.scope_ubuf)
        sign_mask = t.Tensor('int32', (64,), name='native_sign_mask', scope=tik.scope_ubuf)
        zero = t.Tensor('float32', (64,), name='native_zero', scope=tik.scope_ubuf)
        for tensor, value in ((bias, 32767), (parity_mask, 65536), (clear_mask, -65536), (sign_mask, -2147483648)):
            t.vector_dup(64, tensor, value, 1, 1, 8)
        t.vector_dup(64, zero, 0.0, 1, 1, 8)
        t.vector_dup(128, zeros, 0, padded // 128, 1, 8)
        t.vector_dup(128, words, 0, padded // 128, 1, 8)
        if kind in ('rms', 'addrms'):
            sum_dst = t.Tensor('float32', (8,), name='native_sum_dst', scope=tik.scope_ubuf)
            sum_work = t.Tensor('float32', (((padded // 64 + 7) // 8) * 8,), name='native_sum_work', scope=tik.scope_ubuf)
            variance = t.Tensor('float32', (64,), name='native_variance', scope=tik.scope_ubuf)
            inv = t.Tensor('float32', (64,), name='native_inv', scope=tik.scope_ubuf)
            correction = t.Tensor('float32', (64,), name='native_correction', scope=tik.scope_ubuf)
            t.data_move(words, gamma_gm, 0, 1, width // 16, 0, 0)
            unpack_bf16(t, words, gamma, transposed, zeros, padded)
        with t.for_range(0, (rows + cores - 1) // cores) as step:
            row = core + step * cores
            with t.if_scope(row < rows):
                # The padded tail is reset before each load, including channels
                # in the two partial-width blocks (128 and 3200).
                t.vector_dup(128, words, 0, padded // 128, 1, 8)
                t.data_move(words, x_gm[row, 0], 0, 1, width // 16, 0, 0)
                unpack_bf16(t, words, x, transposed, zeros, padded)
                if kind == 'codec':
                    t.data_move(y_gm[row, 0], x, 0, 1, width // 8, 0, 0)
                else:
                    if kind == 'addrms':
                        t.vector_dup(128, words, 0, padded // 128, 1, 8)
                        t.data_move(words, residual_gm[row, 0], 0, 1, width // 16, 0, 0)
                        unpack_bf16(t, words, tmp, transposed, zeros, padded)
                        t.h_add(x, x, tmp)
                        _round_bf16(t, x, x, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    if kind in ('rms', 'addrms'):
                        t.h_mul(tmp, x, x)
                        t.vec_reduce_add(64, sum_dst, tmp, sum_work, padded // 64, 8)
                        raw_sum = t.Scalar('float32', name='native_raw_sum')
                        raw_sum.set_as(sum_dst[0])
                        t.vmuls(8, sum_dst, sum_dst, 1.0 / width, 1, 1, 1, 1, 1)
                        t.vadds(8, sum_dst, sum_dst, eps, 1, 1, 1, 1, 1)
                        value = t.Scalar('float32', name='native_variance_value')
                        value.set_as(sum_dst[0])
                        t.vector_dup(64, variance, value, 1, 1, 8)
                        t.vec_rsqrt(64, inv, variance, 1, 8, 8)
                        initial_inv = t.Scalar('float32', name='native_initial_inv')
                        initial_inv.set_as(inv[0])
                        # 910A vmul/reduce-add saturate FP32 overflow to the
                        # maximum finite value (measured in overflow_stages).
                        # Treat that marker as overflow, preserving the CPU
                        # reference's zero inverse in the tested finite outliers.
                        # Exact legitimately-maximal sums are not certified.
                        with t.if_scope(raw_sum >= 3.4028234663852886e38):
                            t.vector_dup(64, inv, 0.0, 1, 1, 8)
                        with t.else_scope():
                            with t.if_scope(initial_inv > 0.0):
                                for _ in range(2):
                                    t.h_mul(correction, inv, inv)
                                    t.h_mul(correction, correction, variance)
                                    t.vmuls(64, correction, correction, -0.5, 1, 1, 1, 8, 8)
                                    t.vadds(64, correction, correction, 1.5, 1, 1, 1, 8, 8)
                                    t.h_mul(inv, inv, correction)
                        scale = t.Scalar('float32', name='native_scale')
                        scale.set_as(inv[0])
                        t.vmuls(64, dst, x, scale, padded // 64, 1, 1, 8, 8)
                        if qwen:
                            _round_bf16(t, dst, dst, parity, parity_float, bias, parity_mask, clear_mask, padded)
                        t.h_mul(dst, dst, gamma)
                    else:
                        # Packed gate/up input: read the second half directly,
                        # avoiding split-view FP32 casts and contiguous copies.
                        t.vector_dup(128, words, 0, padded // 128, 1, 8)
                        t.data_move(words, x_gm[row, width], 0, 1, width // 16, 0, 0)
                        unpack_bf16(t, words, gamma, transposed, zeros, padded)
                        t.h_abs(tmp, x)
                        t.vmuls(64, tmp, tmp, -1.0, padded // 64, 1, 1, 8, 8)
                        t.h_exp(tmp, tmp)
                        t.vadds(64, dst, tmp, 1.0, padded // 64, 1, 1, 8, 8)
                        t.h_rec(parity_float, dst)
                        for _ in range(2):
                            t.h_mul(parity.reinterpret_cast_to('float32'), dst, parity_float)
                            t.vmuls(64, parity.reinterpret_cast_to('float32'), parity.reinterpret_cast_to('float32'), -1.0, padded // 64, 1, 1, 8, 8)
                            t.vadds(64, parity.reinterpret_cast_to('float32'), parity.reinterpret_cast_to('float32'), 2.0, padded // 64, 1, 1, 8, 8)
                            t.h_mul(parity_float, parity_float, parity.reinterpret_cast_to('float32'))
                        t.h_mul(tmp, tmp, parity_float)
                        for i in range(padded // 64):
                            mask = t.vcmp_ge(64, x[i * 64], zero, 1, 1)
                            t.vsel(64, 0, dst[i * 64], mask, parity_float[i * 64], tmp[i * 64], 1, 1, 1, 1, 8, 8, 8)
                        t.h_mul(dst, dst, x)
                        _round_bf16(t, dst, dst, parity, parity_float, bias, parity_mask, clear_mask, padded)
                        t.h_mul(dst, dst, gamma)
                    _round_bf16(t, dst, dst, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    if kind in ('rms', 'addrms'):
                        _fix_product_zero(t, dst, x, gamma, parity, parity_float, sign_mask, zero, padded)
                    # SwiGLU retains the native NPU multiply's zero behavior;
                    # CPU SiLU differs at extreme exp-overflow inputs and zeros.
                    # All low words are zero here, so pack's discarded words
                    # do not dirty the zero source subsequently used by unpack.
                    pack_bf16(t, dst, words, transposed, zeros, padded)
                    t.data_move(y_gm[row, 0], words, 0, 1, width // 16, 0, 0)
                pack_bf16(t, x, words, transposed, zeros, padded)
                t.data_move(updated_gm[row, 0], words, 0, 1, width // 16, 0, 0)
    work = os.path.realpath(tempfile.mkdtemp(prefix='native58_build_', dir=ROOT))
    if Path(work).parent != ROOT.resolve() or not Path(work).name.startswith('native58_build_'):
        raise RuntimeError('Unexpected native build directory')
    old = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name, inputs=[x_gm, gamma_gm, residual_gm], outputs=[y_gm, updated_gm])
        src = Path(work) / 'kernel_meta' / (name + '.o')
        assert src.exists() and src.resolve() == src.absolute()
        target.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'), obj.with_suffix('.json'))
        shutil.copyfile(src, obj)
    finally:
        os.chdir(old)
        if Path(work).resolve() == Path(work).absolute() and Path(work).parent == ROOT.resolve():
            shutil.rmtree(work)
    meta = json.loads(obj.with_suffix('.json').read_text())
    assert meta['blockDim'] == cores
    return str(obj), name + '__kernel0', cores


def native_launch(kind, x, weight=None, residual=None, eps=1e-6, qwen=True):
    assert kind in ('codec', 'rms', 'addrms', 'swiglu')
    assert x.dtype == torch.bfloat16 and x.device.type in ('npu', 'privateuseone') and x.is_contiguous()
    import torch_npu
    assert torch_npu.get_npu_format(x) in (0, 2), 'Native codec requires dense ND/NCHW memory'
    width = x.shape[-1] // 2 if kind == 'swiglu' else x.shape[-1]
    if kind == 'swiglu':
        assert x.shape[-1] == 2 * width, 'Packed gate/up halves must be equal'
    if kind in ('rms', 'addrms'):
        assert weight is not None and weight.numel() == width
    if kind == 'addrms':
        assert residual is not None and residual.shape == x.shape
    rows = x.numel() // x.shape[-1]
    assert 64 <= width <= 8192 and width % 64 == 0 and rows > 0
    weight = x if weight is None else weight
    residual = x if residual is None else residual
    assert weight.dtype == residual.dtype == x.dtype and weight.device == residual.device == x.device
    assert weight.is_contiguous() and residual.is_contiguous()
    assert torch_npu.get_npu_format(weight) in (0, 2) and torch_npu.get_npu_format(residual) in (0, 2)
    cores = min(core_budget(), rows)
    key = (kind, rows, width, bool(qwen), float(eps), cores)
    info = _COMPILED.get(key)
    if info is None:
        with (KERNELS / '.compile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            info = _build(kind, rows, width, qwen, eps, cores)
        _COMPILED[key] = info
    obj, symbol, cores = info
    if obj not in _HANDLES:
        h = library().fa910a_register(obj.encode(), symbol.encode())
        if not h:
            raise RuntimeError('Native vector registration failed')
        _HANDLES[obj] = h
    shape = tuple(x.shape[:-1]) + (width,)
    y = torch.empty(shape, dtype=torch.float32 if kind == 'codec' else x.dtype, device=x.device)
    updated = torch.empty(shape, dtype=x.dtype, device=x.device)
    rc = library().fa910a_launch(_HANDLES[obj], symbol.encode(), cores,
                                 x.data_ptr(), weight.data_ptr(), residual.data_ptr(), y.data_ptr(), updated.data_ptr(),
                                 ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Native vector launch failed rc=%s' % rc)
    return y, updated
