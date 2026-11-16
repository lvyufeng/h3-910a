"""Self-developed BF16-GM RoPE for Ascend910A, explicit intermediate rounding.

The kernel batches every token's heads in one UB tile, preserving BNSD caller
shape through BSND dense memory. It reuses the proven integer BF16 codec/RNE.
Native-style FP32 products and HF-style rounded BF16 products are separate cache
contracts. No production patch or global precision flags are changed.
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

ROOT = Path('/tmp/h3_round60_rope01')
KERNELS = ROOT / 'kernels'
_HASH = hashlib.sha256((Path(__file__).read_bytes() + Path('/tmp/h3_round58_vec01/fused_vector58.py').read_bytes()
                        + Path('/tmp/h3_round58_vec01/native_bf16_vector58.py').read_bytes())).hexdigest()[:12]
_CACHE = {}
_HANDLES = {}


def _build(tokens, heads, hf_round, cores):
    from tbe import tik
    import tbe.common.platform as platform
    platform.set_current_compile_soc_info('Ascend910A')
    d, width = 128, heads * 128
    padded = ((width + 255) // 256) * 256
    name = f'rope60_n{tokens}_h{heads}_d128_hf{int(hf_round)}_c{cores}_{_HASH}'
    obj = KERNELS / name / (name + '.o')
    if obj.exists():
        meta = json.loads(obj.with_suffix('.json').read_text()); assert meta['blockDim'] == cores
        return str(obj), name+'__kernel0', cores
    t = tik.Tik(disable_debug=True)
    xgm = t.Tensor('uint16', (tokens, width), name='rope_xgm', scope=tik.scope_gm)
    cgm = t.Tensor('uint16', (tokens, d), name='rope_cgm', scope=tik.scope_gm)
    sgm = t.Tensor('uint16', (tokens, d), name='rope_sgm', scope=tik.scope_gm)
    ygm = t.Tensor('uint16', (tokens, width), name='rope_ygm', scope=tik.scope_gm)
    agm = t.Tensor('uint16', (tokens, width), name='rope_agm', scope=tik.scope_gm)
    with t.for_range(0, cores, block_num=cores) as core:
        words = t.Tensor('uint16', (padded,), name='rope_words', scope=tik.scope_ubuf)
        transposed = t.Tensor('uint16', (padded,), name='rope_transposed', scope=tik.scope_ubuf)
        zeros = t.Tensor('uint16', (padded,), name='rope_zero_words', scope=tik.scope_ubuf)
        x = t.Tensor('float32', (padded,), name='rope_x', scope=tik.scope_ubuf)
        cos = t.Tensor('float32', (padded,), name='rope_cos', scope=tik.scope_ubuf)
        sin = t.Tensor('float32', (padded,), name='rope_sin', scope=tik.scope_ubuf)
        rotated = t.Tensor('float32', (padded,), name='rope_rotated', scope=tik.scope_ubuf)
        out = t.Tensor('float32', (padded,), name='rope_out', scope=tik.scope_ubuf)
        parity = t.Tensor('int32', (padded,), name='rope_parity', scope=tik.scope_ubuf)
        parity_float = t.Tensor('float32', (padded,), name='rope_parity_float', scope=tik.scope_ubuf)
        bias = t.Tensor('int32', (64,), name='rope_bias', scope=tik.scope_ubuf)
        parity_mask = t.Tensor('int32', (64,), name='rope_parity_mask', scope=tik.scope_ubuf)
        clear_mask = t.Tensor('int32', (64,), name='rope_clear_mask', scope=tik.scope_ubuf)
        t.vector_dup(64, bias, 32767, 1, 1, 8)
        t.vector_dup(64, parity_mask, 65536, 1, 1, 8)
        t.vector_dup(64, clear_mask, -65536, 1, 1, 8)
        t.vector_dup(128, zeros, 0, padded//128, 1, 8)
        with t.for_range(0, (tokens+cores-1)//cores) as step:
            row = core + step * cores
            with t.if_scope(row < tokens):
                t.vector_dup(128, words, 0, padded//128, 1, 8)
                t.data_move(words, xgm[row, 0], 0, 1, width//16, 0, 0)
                unpack_bf16(t, words, x, transposed, zeros, padded)
                t.vector_dup(128, words, 0, padded//128, 1, 8)
                t.data_move(words, cgm[row, 0], 0, 1, d//16, 0, 0)
                unpack_bf16(t, words, cos, transposed, zeros, 256)
                t.vector_dup(128, words, 0, padded//128, 1, 8)
                t.data_move(words, sgm[row, 0], 0, 1, d//16, 0, 0)
                unpack_bf16(t, words, sin, transposed, zeros, 256)
                for head in range(1, heads):
                    t.data_move(cos[head*d], cos, 0, 1, d//8, 0, 0)
                    t.data_move(sin[head*d], sin, 0, 1, d//8, 0, 0)
                t.vector_dup(64, rotated, 0.0, padded//64, 1, 8)
                for head in range(heads):
                    t.data_move(rotated[head*d], x[head*d+d//2], 0, 1, d//16, 0, 0)
                    t.data_move(rotated[head*d+d//2], x[head*d], 0, 1, d//16, 0, 0)
                t.vmuls(64, rotated, rotated, -1.0, heads, 1, 1, d//8, d//8)
                t.h_mul(out, x, cos)
                t.h_mul(rotated, rotated, sin)
                if hf_round:
                    _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                    _round_bf16(t, rotated, rotated, parity, parity_float, bias, parity_mask, clear_mask, padded)
                t.h_add(out, out, rotated)
                _round_bf16(t, out, out, parity, parity_float, bias, parity_mask, clear_mask, padded)
                pack_bf16(t, out, words, transposed, zeros, padded)
                t.data_move(ygm[row, 0], words, 0, 1, width//16, 0, 0)
                # Existing five-pointer ABI includes an auxiliary immutable
                # input copy; no scratch buffer is returned/reused as output.
                pack_bf16(t, x, words, transposed, zeros, padded)
                t.data_move(agm[row, 0], words, 0, 1, width//16, 0, 0)
    work = os.path.realpath(tempfile.mkdtemp(prefix='rope60_build_', dir=ROOT))
    if Path(work).parent != ROOT.resolve() or not Path(work).name.startswith('rope60_build_'):
        raise RuntimeError('Unexpected RoPE build directory')
    old = os.getcwd()
    try:
        os.chdir(work)
        t.BuildCCE(kernel_name=name, inputs=[xgm, cgm, sgm], outputs=[ygm, agm])
        src = Path(work)/'kernel_meta'/(name+'.o')
        assert src.exists() and src.resolve() == src.absolute()
        obj.parent.mkdir(exist_ok=False)
        shutil.copyfile(src.with_suffix('.json'), obj.with_suffix('.json')); shutil.copyfile(src, obj)
    finally:
        os.chdir(old)
        if Path(work).resolve() == Path(work).absolute() and Path(work).parent == ROOT.resolve():
            shutil.rmtree(work)
    meta = json.loads(obj.with_suffix('.json').read_text()); assert meta['blockDim'] == cores
    return str(obj), name+'__kernel0', cores


def rope(x, cos, sin, hf_round=False):
    assert x.ndim == 4 and x.shape[0] == 1 and x.shape[-1] == 128 and x.shape[1] in (1, 8)
    assert x.dtype == cos.dtype == sin.dtype == torch.bfloat16
    assert not (torch.is_grad_enabled() and any(t.requires_grad for t in (x, cos, sin))), 'Inference-only custom RoPE'
    assert x.device == cos.device == sin.device and x.device.type in ('npu', 'privateuseone')
    tokens, heads = x.shape[2], x.shape[1]
    assert 0 < tokens <= 4096 and cos.shape == sin.shape == (1, tokens, 128)
    dense = x.transpose(1, 2).contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    assert all(torch_npu.get_npu_format(t) in (0, 2) for t in (dense, cos, sin))
    cores = min(core_budget(), tokens)
    key = (tokens, heads, bool(hf_round), cores)
    info = _CACHE.get(key)
    if info is None:
        KERNELS.mkdir(exist_ok=True)
        with (KERNELS/'.compile.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            info = _build(tokens, heads, hf_round, cores)
        _CACHE[key] = info
    obj, symbol, cores = info
    if obj not in _HANDLES:
        h = library().fa910a_register(obj.encode(), symbol.encode())
        if not h:
            raise RuntimeError('RoPE registration failed')
        _HANDLES[obj] = h
    out, auxiliary = torch.empty_like(dense), torch.empty_like(dense)
    rc = library().fa910a_launch(_HANDLES[obj], symbol.encode(), cores, dense.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                                 out.data_ptr(), auxiliary.data_ptr(), ctypes.c_void_p(torch.npu.current_stream().npu_stream))
    if rc:
        raise RuntimeError('RoPE launch failed rc=%s' % rc)
    return out.transpose(1, 2)
