"""Exact native-half isfinite/where-zero TBE kernel for isolated VAE attention.

Preserve every finite half word, including signed zero and subnormals. Select
positive zero for both infinities and every NaN payload. No FP32 conversion,
score clipping or attention arithmetic changes are introduced.
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
import torch_npu
from tbe import tik
import tbe.common.platform as platform
from vllm_omni.diffusion.attention.backends import fa910a as fa

ROOT = Path('/tmp/h3_round63_decode01/sanitize_kernels')
SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
CACHE = {}


def build(elements, cores):
    assert elements > 0 and elements % 128 == 0
    chunk = 4096
    tasks = (elements + chunk - 1) // chunk
    cores = min(cores, tasks)
    name = f'sanitize63_e{elements}_c{cores}_{SOURCE}'
    ROOT.mkdir(parents=True, exist_ok=True)
    obj = ROOT / name / (name + '.o')
    with (ROOT / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.exists():
            assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
            return str(obj), name + '__kernel0', cores
        platform.set_current_compile_soc_info('Ascend910A')
        t = tik.Tik(disable_debug=True)
        xgm = t.Tensor('float16', (elements,), name='finite_xgm', scope=tik.scope_gm)
        aux = [t.Tensor('float16', (16,), name='finite_aux%d' % i, scope=tik.scope_gm) for i in range(3)]
        ygm = t.Tensor('float16', (elements,), name='finite_ygm', scope=tik.scope_gm)
        with t.for_range(0, cores, block_num=cores) as core:
            x = t.Tensor('float16', (chunk,), name='finite_x', scope=tik.scope_ubuf)
            absolute = t.Tensor('float16', (chunk,), name='finite_abs', scope=tik.scope_ubuf)
            bound = t.Tensor('float16', (128,), name='finite_bound', scope=tik.scope_ubuf)
            zero = t.Tensor('float16', (128,), name='finite_zero', scope=tik.scope_ubuf)
            t.vector_dup(128, bound, 65504., 1, 1, 8)
            t.vector_dup(128, zero, 0., 1, 1, 8)
            def process(offset, count):
                t.data_move(x, xgm[offset], 0, 1, count // 16, 0, 0)
                t.vabs(128, absolute, x, count // 128, 1, 1, 8, 8)
                with t.for_range(0, count // 128) as i:
                    mask = t.vcmp_le(128, absolute[i * 128], bound, 1, 1)
                    t.vsel(128, 0, x[i * 128], mask, x[i * 128], zero, 1, 1, 1, 1, 8, 8, 8)
                t.data_move(ygm[offset], x, 0, 1, count // 16, 0, 0)
            with t.for_range(0, (tasks + cores - 1) // cores) as step:
                task = core + step * cores
                with t.if_scope(task < tasks):
                    offset = task * chunk
                    with t.if_scope(offset + chunk <= elements):
                        process(offset, chunk)
                    if elements % chunk:
                        with t.else_scope():
                            process(offset, elements % chunk)
        work = Path(tempfile.mkdtemp(prefix='sanitize63_build_', dir=ROOT)).resolve(strict=True)
        if work.parent != ROOT.resolve() or not work.name.startswith('sanitize63_build_'):
            raise RuntimeError('Unexpected sanitizer build path')
        previous = os.getcwd()
        try:
            os.chdir(work)
            t.BuildCCE(kernel_name=name, inputs=[xgm, *aux], outputs=[ygm])
            source = work / 'kernel_meta' / (name + '.o')
            assert source.is_file()
            obj.parent.mkdir(exist_ok=False)
            shutil.copyfile(source.with_suffix('.json'), obj.with_suffix('.json'))
            shutil.copyfile(source, obj)
        finally:
            os.chdir(previous)
            if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('sanitize63_build_'):
                raise RuntimeError('Unexpected sanitizer cleanup target')
            shutil.rmtree(work)
        assert json.loads(obj.with_suffix('.json').read_text())['blockDim'] == cores
        return str(obj), name + '__kernel0', cores


def sanitize(x):
    if x.dtype != torch.float16 or x.device.type not in ('npu', 'privateuseone') or x.numel() % 128 or x.numel() == 0:
        raise ValueError('Sanitizer requires nonempty NPU half with a 128-element-aligned length')
    dense = x.contiguous()
    if torch_npu.get_npu_format(dense) not in (0, 2):
        dense = torch_npu.npu_format_cast(dense, 0)
    if dense.data_ptr() % 32:
        dense = dense.clone()
    elements = dense.numel()
    device = torch.npu.current_device()
    if dense.device.index != device:
        raise ValueError('Input device must be the current launch device')
    key = (elements, device)
    if key not in CACHE:
        obj, name, cores = build(elements, fa._runtime_core_budget())
        handle = fa._lib().fa910a_register(obj.encode(), name.encode())
        if not handle:
            raise RuntimeError('Sanitizer registration failed')
        CACHE[key] = (handle, name, cores)
    handle, name, cores = CACHE[key]
    out = torch.empty_like(dense)
    rc = fa._lib().fa910a_launch(handle, name.encode(), cores, dense.data_ptr(), dense.data_ptr(), dense.data_ptr(),
                                 dense.data_ptr(), out.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('Sanitizer launch failed: %d' % rc)
    return out
