"""910A integer-codec BF16 activation expansion, preserving every source bit.

This contains no arithmetic, rounding, clipping, weights or GEMM. It exposes
only aligned, contiguous dense activations; actual native cast output bit/layout
and subsequent projection/full-tensor identity decide acceptance. Noncontiguous
or non-ND input is rejected, not silently reformatted into a different operand.
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
from native_bf16_vector58 import unpack_bf16
from vllm_omni.diffusion.attention.backends import fa910a as fa

ROOT = Path('/tmp/h3_round69_cast01')
KERNELS = ROOT / 'kernels'
SOURCE = hashlib.sha256(Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes()).hexdigest()[:12]
CACHE = {}


def build(elements, cores):
    from tbe import tik
    import tbe.common.platform as platform
    assert elements > 0 and elements % 128 == 0
    chunk = 4096
    tasks = (elements + chunk - 1) // chunk
    cores = min(cores, tasks)
    name = f'cast69_e{elements}_c{cores}_{SOURCE}'
    KERNELS.mkdir(exist_ok=True)
    obj = KERNELS / name / (name + '.o')
    with (KERNELS / '.compile.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.exists():
            assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
            return str(obj), name + '__kernel0', cores
        platform.set_current_compile_soc_info('Ascend910A')
        t = tik.Tik(disable_debug=True)
        xgm = t.Tensor('uint16', (elements,), name='cast_xgm', scope=tik.scope_gm)
        auxgm = [t.Tensor('uint16', (16,), name='cast_aux%d' % i, scope=tik.scope_gm) for i in range(3)]
        ygm = t.Tensor('float32', (elements,), name='cast_ygm', scope=tik.scope_gm)
        with t.for_range(0, cores, block_num=cores) as core:
            words = t.Tensor('uint16', (chunk,), name='cast_words', scope=tik.scope_ubuf)
            transposed = t.Tensor('uint16', (chunk,), name='cast_transposed', scope=tik.scope_ubuf)
            zeros = t.Tensor('uint16', (chunk,), name='cast_zeros', scope=tik.scope_ubuf)
            values = t.Tensor('float32', (chunk,), name='cast_values', scope=tik.scope_ubuf)
            t.vector_dup(128, zeros, 0, chunk // 128, 1, 8)
            def process(offset, count):
                padded = ((count + 255) // 256) * 256
                t.vector_dup(128, words, 0, padded // 128, 1, 8)
                t.data_move(words, xgm[offset], 0, 1, count // 16, 0, 0)
                unpack_bf16(t, words, values, transposed, zeros, padded)
                # Only real elements go to GM, never padded tails.
                t.data_move(ygm[offset], values, 0, 1, count // 8, 0, 0)
            with t.for_range(0, (tasks + cores - 1) // cores) as step:
                task = core + step * cores
                with t.if_scope(task < tasks):
                    offset = task * chunk
                    with t.if_scope(offset + chunk <= elements):
                        process(offset, chunk)
                    if elements % chunk:
                        with t.else_scope():
                            process(offset, elements % chunk)
        work = Path(tempfile.mkdtemp(prefix='cast69_build_', dir=ROOT)).resolve(strict=True)
        if work.parent != ROOT.resolve() or not work.name.startswith('cast69_build_'):
            raise RuntimeError('Unexpected BF16-cast build directory')
        previous = os.getcwd()
        try:
            os.chdir(work)
            t.BuildCCE(kernel_name=name, inputs=[xgm, *auxgm], outputs=[ygm])
            source = work / 'kernel_meta' / (name + '.o')
            assert source.is_file()
            obj.parent.mkdir(exist_ok=False)
            shutil.copyfile(source.with_suffix('.json'), obj.with_suffix('.json'))
            shutil.copyfile(source, obj)
        finally:
            os.chdir(previous)
            if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('cast69_build_'):
                raise RuntimeError('Unexpected BF16-cast cleanup directory')
            shutil.rmtree(work)
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores


def expand_activation(x):
    if (x.dtype != torch.bfloat16 or x.device.type not in ('npu', 'privateuseone') or not x.is_contiguous()
            or x.numel() == 0 or x.numel() % 128 or x.device.index != torch.npu.current_device()
            or x.data_ptr() % 32 or torch_npu.get_npu_format(x) not in (0, 2)):
        raise ValueError('BF16 expansion requires current-device aligned contiguous dense activation with128-element length alignment')
    elements = x.numel()
    key = (elements, x.device.index)
    if key not in CACHE:
        obj, symbol, cores = build(elements, fa._runtime_core_budget())
        handle = fa._lib().fa910a_register(obj.encode(), symbol.encode())
        if not handle:
            raise RuntimeError('BF16 expansion registration failed')
        CACHE[key] = (handle, symbol, cores)
    handle, symbol, cores = CACHE[key]
    out = torch.empty_like(x, dtype=torch.float32)
    if out.data_ptr() % 32 or torch_npu.get_npu_format(out) not in (0, 2) or out.stride() != x.stride():
        raise RuntimeError('Output does not preserve required dense activation layout')
    rc = fa._lib().fa910a_launch(handle, symbol.encode(), cores, x.data_ptr(), x.data_ptr(), x.data_ptr(), x.data_ptr(),
                                out.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('BF16 expansion launch failed: %d' % rc)
    return out
