"""910A BF16->FP32 square prefix, optionally with BF16 residual update.

Native mean/reduction, epsilon and rsqrt are NOT implemented here. Products and
optional residual add have to match the actual native FP32 operations on bounded
finite fixtures and every actual Norm. Overflow/NaN behavior is not assumed
identical. No signed-zero correction or Newton step is used.
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
from fused_vector58 import _round_bf16
from vllm_omni.diffusion.attention.backends import fa910a as fa

ROOT = Path('/tmp/h3_round68_prefix01')
KERNELS = ROOT / 'kernels'
SOURCE = hashlib.sha256(Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()
                        + Path('/tmp/h3_round58_vec01/fused_vector58.py').read_bytes()).hexdigest()[:12]
CACHE = {}


def build(rows, width, with_residual, cores):
    from tbe import tik
    import tbe.common.platform as platform
    padded = ((width + 255) // 256) * 256
    name = f'norm_prefix68_r{rows}_d{width}_a{int(with_residual)}_c{cores}_{SOURCE}'
    KERNELS.mkdir(exist_ok=True)
    obj = KERNELS / name / (name + '.o')
    with (KERNELS / '.compile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.exists():
            assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
            return str(obj), name + '__kernel0', cores
        platform.set_current_compile_soc_info('Ascend910A')
        t = tik.Tik(disable_debug=True)
        xgm = t.Tensor('uint16', (rows, width), name='prefix_xgm', scope=tik.scope_gm)
        rgm = t.Tensor('uint16', (rows, width), name='prefix_rgm', scope=tik.scope_gm)
        auxgm = t.Tensor('uint16', (16,), name='prefix_auxgm', scope=tik.scope_gm)
        sgm = t.Tensor('float32', (rows, width), name='prefix_sgm', scope=tik.scope_gm)
        ugm = t.Tensor('uint16', (rows, width) if with_residual else (16,), name='prefix_ugm', scope=tik.scope_gm)
        with t.for_range(0, cores, block_num=cores) as core:
            words = t.Tensor('uint16', (padded,), name='prefix_words', scope=tik.scope_ubuf)
            transposed = t.Tensor('uint16', (padded,), name='prefix_transposed', scope=tik.scope_ubuf)
            zeros = t.Tensor('uint16', (padded,), name='prefix_zeros', scope=tik.scope_ubuf)
            x = t.Tensor('float32', (padded,), name='prefix_x', scope=tik.scope_ubuf)
            other = t.Tensor('float32', (padded,), name='prefix_other', scope=tik.scope_ubuf)
            square = t.Tensor('float32', (padded,), name='prefix_square', scope=tik.scope_ubuf)
            if with_residual:
                parity = t.Tensor('int32', (padded,), name='prefix_parity', scope=tik.scope_ubuf)
                parity_float = t.Tensor('float32', (padded,), name='prefix_parity_float', scope=tik.scope_ubuf)
                bias = t.Tensor('int32', (64,), name='prefix_bias', scope=tik.scope_ubuf)
                parity_mask = t.Tensor('int32', (64,), name='prefix_parity_mask', scope=tik.scope_ubuf)
                clear_mask = t.Tensor('int32', (64,), name='prefix_clear_mask', scope=tik.scope_ubuf)
                for tensor, value in ((bias, 32767), (parity_mask, 65536), (clear_mask, -65536)):
                    t.vector_dup(64, tensor, value, 1, 1, 8)
            t.vector_dup(128, zeros, 0, padded // 128, 1, 8)
            with t.for_range(0, (rows + cores - 1) // cores) as step:
                row = core + step * cores
                with t.if_scope(row < rows):
                    t.vector_dup(128, words, 0, padded // 128, 1, 8)
                    t.data_move(words, xgm[row, 0], 0, 1, width // 16, 0, 0)
                    unpack_bf16(t, words, x, transposed, zeros, padded)
                    if with_residual:
                        t.vector_dup(128, words, 0, padded // 128, 1, 8)
                        t.data_move(words, rgm[row, 0], 0, 1, width // 16, 0, 0)
                        unpack_bf16(t, words, other, transposed, zeros, padded)
                        t.h_add(x, other, x)
                        _round_bf16(t, x, x, parity, parity_float, bias, parity_mask, clear_mask, padded)
                        pack_bf16(t, x, words, transposed, zeros, padded)
                        t.data_move(ugm[row, 0], words, 0, 1, width // 16, 0, 0)
                    # Keep ALL FP32 product bits; do not round square to BF16.
                    t.h_mul(square, x, x)
                    t.data_move(sgm[row, 0], square, 0, 1, width // 8, 0, 0)
        work = Path(tempfile.mkdtemp(prefix='norm_prefix68_build_', dir=ROOT)).resolve(strict=True)
        if work.parent != ROOT.resolve() or not work.name.startswith('norm_prefix68_build_'):
            raise RuntimeError('Unexpected Norm-prefix build path')
        previous = os.getcwd()
        try:
            os.chdir(work)
            t.BuildCCE(kernel_name=name, inputs=[xgm, rgm, auxgm], outputs=[sgm, ugm])
            source = work / 'kernel_meta' / (name + '.o')
            assert source.is_file()
            obj.parent.mkdir(exist_ok=False)
            shutil.copyfile(source.with_suffix('.json'), obj.with_suffix('.json'))
            shutil.copyfile(source, obj)
        finally:
            os.chdir(previous)
            if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('norm_prefix68_build_'):
                raise RuntimeError('Unexpected Norm-prefix cleanup path')
            shutil.rmtree(work)
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores


def norm_prefix(x, residual=None):
    if (x.dtype != torch.bfloat16 or x.device.type not in ('npu', 'privateuseone') or x.numel() == 0
            or x.shape[-1] not in (128, 5120) or x.device.index != torch.npu.current_device()):
        raise ValueError('Unsupported actual Norm-prefix input')
    if residual is not None and (residual.shape != x.shape or residual.device != x.device or residual.dtype != x.dtype):
        raise ValueError('Residual must match original BF16 input shape/device/dtype')
    dense = x.contiguous()
    other = dense if residual is None else residual.contiguous()
    if any(value.data_ptr() % 32 or torch_npu.get_npu_format(value) not in (0, 2) for value in (dense, other)):
        raise ValueError('Norm-prefix requires aligned dense ND/NCHW storage')
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    active = residual is not None
    key = (rows, width, active, x.device.index)
    if key not in CACHE:
        obj, symbol, cores = build(rows, width, active, min(fa._runtime_core_budget(), rows))
        handle = fa._lib().fa910a_register(obj.encode(), symbol.encode())
        if not handle:
            raise RuntimeError('Norm-prefix registration failed')
        CACHE[key] = (handle, symbol, cores)
    handle, symbol, cores = CACHE[key]
    squares = torch.empty_like(dense, dtype=torch.float32)
    updated = torch.empty_like(dense) if active else dense
    rc = fa._lib().fa910a_launch(handle, symbol.encode(), cores, dense.data_ptr(), other.data_ptr(), dense.data_ptr(),
                                squares.data_ptr(), updated.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Norm-prefix launch failed: %d' % rc)
    return squares, updated
