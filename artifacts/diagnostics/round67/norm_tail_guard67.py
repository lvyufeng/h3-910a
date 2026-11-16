"""910A BF16-GM Norm pointwise tail using unmodified native FP32 inverse-RMS.

No reduction, rsqrt, residual add, Newton step or signed-zero correction. This is
an isolated diagnostic kernel, not a production dispatch or arbitrary exceptional
input equivalence promise. Both explicit BF16 round boundaries use the already
validated finite-RNE integer codec. Real same-input and full-stack checks decide
acceptance, not that codec claim alone.
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
from torch.nn import functional as F
sys.path.insert(0, '/tmp/h3_round58_vec01')
from native_bf16_vector58 import unpack_bf16, pack_bf16
from fused_vector58 import _round_bf16
from vllm_omni.diffusion.attention.backends import fa910a as fa

ROOT = Path('/tmp/h3_round67_norm01')
KERNELS = ROOT / 'kernels'
SOURCE = hashlib.sha256(Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()
                        + Path('/tmp/h3_round58_vec01/fused_vector58.py').read_bytes()).hexdigest()[:12]
CACHE = {}


def build(rows, width, cores):
    from tbe import tik
    import tbe.common.platform as platform
    padded = ((width + 255) // 256) * 256
    inv_rows = ((rows + 7) // 8) * 8
    name = f'norm_tail67_r{rows}_d{width}_c{cores}_{SOURCE}'
    KERNELS.mkdir(exist_ok=True)
    obj = KERNELS / name / (name + '.o')
    with (KERNELS / '.compile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.exists():
            assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
            return str(obj), name + '__kernel0', cores
        platform.set_current_compile_soc_info('Ascend910A')
        t = tik.Tik(disable_debug=True)
        xgm = t.Tensor('uint16', (rows, width), name='tail_xgm', scope=tik.scope_gm)
        ggm = t.Tensor('uint16', (width,), name='tail_ggm', scope=tik.scope_gm)
        igm = t.Tensor('float32', (inv_rows,), name='tail_igm', scope=tik.scope_gm)
        ygm = t.Tensor('uint16', (rows, width), name='tail_ygm', scope=tik.scope_gm)
        auxgm = t.Tensor('uint16', (16,), name='tail_auxgm', scope=tik.scope_gm)
        with t.for_range(0, cores, block_num=cores) as core:
            words = t.Tensor('uint16', (padded,), name='tail_words', scope=tik.scope_ubuf)
            transposed = t.Tensor('uint16', (padded,), name='tail_transposed', scope=tik.scope_ubuf)
            zeros = t.Tensor('uint16', (padded,), name='tail_zeros', scope=tik.scope_ubuf)
            x = t.Tensor('float32', (padded,), name='tail_x', scope=tik.scope_ubuf)
            gamma = t.Tensor('float32', (padded,), name='tail_gamma', scope=tik.scope_ubuf)
            out = t.Tensor('float32', (padded,), name='tail_out', scope=tik.scope_ubuf)
            parity = t.Tensor('int32', (padded,), name='tail_parity', scope=tik.scope_ubuf)
            parity_float = t.Tensor('float32', (padded,), name='tail_parity_float', scope=tik.scope_ubuf)
            bias = t.Tensor('int32', (64,), name='tail_bias', scope=tik.scope_ubuf)
            parity_mask = t.Tensor('int32', (64,), name='tail_parity_mask', scope=tik.scope_ubuf)
            clear_mask = t.Tensor('int32', (64,), name='tail_clear_mask', scope=tik.scope_ubuf)
            inv = t.Tensor('float32', (8,), name='tail_inv', scope=tik.scope_ubuf)
            for tensor, value in ((bias, 32767), (parity_mask, 65536), (clear_mask, -65536)):
                t.vector_dup(64, tensor, value, 1, 1, 8)
            t.vector_dup(128, zeros, 0, padded // 128, 1, 8)
            t.vector_dup(128, words, 0, padded // 128, 1, 8)
            t.data_move(words, ggm, 0, 1, width // 16, 0, 0)
            unpack_bf16(t, words, gamma, transposed, zeros, padded)
            with t.for_range(0, (rows + cores - 1) // cores) as step:
                row = core + step * cores
                with t.if_scope(row < rows):
                    # 32-byte inverse blocks are fully allocated, including
                    # non-multiple-of8 row counts such as2043. Never overread.
                    t.data_move(inv, igm[(row // 8) * 8], 0, 1, 1, 0, 0)
                    scale = t.Scalar('float32', name='tail_scale')
                    scale.set_as(inv[row % 8])
                    t.vector_dup(128, words, 0, padded // 128, 1, 8)
                    t.data_move(words, xgm[row, 0], 0, 1, width // 16, 0, 0)
                    unpack_bf16(t, words, x, transposed, zeros, padded)
                    t.vmuls(64, out, x, scale, padded // 64, 1, 1, 8, 8)
                    # normalized.to(BF16).float(): round and clear low words.
                    _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    t.h_mul(out, out, gamma)
                    _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    pack_bf16(t, out, words, transposed, zeros, padded)
                    t.data_move(ygm[row, 0], words, 0, 1, width // 16, 0, 0)
        work = Path(tempfile.mkdtemp(prefix='norm_tail67_build_', dir=ROOT)).resolve(strict=True)
        if work.parent != ROOT.resolve() or not work.name.startswith('norm_tail67_build_'):
            raise RuntimeError('Unexpected Norm-tail build target')
        previous = os.getcwd()
        try:
            os.chdir(work)
            t.BuildCCE(kernel_name=name, inputs=[xgm, ggm, igm], outputs=[ygm, auxgm])
            source = work / 'kernel_meta' / (name + '.o')
            assert source.is_file()
            obj.parent.mkdir(exist_ok=False)
            shutil.copyfile(source.with_suffix('.json'), obj.with_suffix('.json'))
            shutil.copyfile(source, obj)
        finally:
            os.chdir(previous)
            if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('norm_tail67_build_'):
                raise RuntimeError('Unexpected Norm-tail cleanup target')
            shutil.rmtree(work)
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores


def norm_tail(x, gamma, inverse_rms):
    if (x.dtype != torch.bfloat16 or gamma.dtype != x.dtype or inverse_rms.dtype != torch.float32
            or x.device.type not in ('npu', 'privateuseone') or x.device != gamma.device or x.device != inverse_rms.device
            or x.shape[-1] not in (128, 5120) or gamma.shape != (x.shape[-1],)
            or x.numel() == 0 or tuple(inverse_rms.shape) != tuple(x.shape[:-1]) + (1,)):
        raise ValueError('Unsupported actual text Norm-tail shape/dtype/device')
    if x.device.index != torch.npu.current_device():
        raise ValueError('Input must use the current launch device')
    dense, gamma, inverse_rms = x.contiguous(), gamma.contiguous(), inverse_rms.contiguous()
    if any(torch_npu.get_npu_format(value) not in (0, 2) for value in (dense, gamma, inverse_rms)):
        raise ValueError('Norm-tail requires dense ND/NCHW storage')
    rows, width = x.numel() // x.shape[-1], x.shape[-1]
    # Required for safe aligned32-byte GM reads. Values are not transformed.
    inv = F.pad(inverse_rms.reshape(-1), (0, (-rows) % 8)).contiguous()
    if any(value.data_ptr() % 32 for value in (dense, gamma, inv)):
        raise ValueError('Norm-tail requires32-byte aligned GM addresses, including contiguous views')
    key = (rows, width, x.device.index)
    if key not in CACHE:
        obj, symbol, cores = build(rows, width, min(fa._runtime_core_budget(), rows))
        handle = fa._lib().fa910a_register(obj.encode(), symbol.encode())
        if not handle:
            raise RuntimeError('Norm-tail kernel registration failed')
        CACHE[key] = (handle, symbol, cores)
    handle, symbol, cores = CACHE[key]
    result = torch.empty_like(dense)
    rc = fa._lib().fa910a_launch(handle, symbol.encode(), cores, dense.data_ptr(), gamma.data_ptr(), inv.data_ptr(),
                                result.data_ptr(), result.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Norm-tail launch failed: %d' % rc)
    return result
