"""910A TIK: fuse actual VAE BSND Q/K/interleaved-V -> padded NZ + Q scale.

Both pack and unpack use the existing five-GM-pointer launcher ABI. One combined
output holds Q/K/V NZ consecutively, so three inputs + one reserved input + one
output suffice. No shared production module or launcher binary is modified.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from tbe import tik
import tbe.common.platform as platform

ROOT = Path('/tmp/h3_round63_decode01/pack_kernels')
SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def build(kind, n, padded, heads, cores, v_row_stride=None, v_head_stride=None):
    assert kind in ('pack', 'unpack') and 0 < n <= padded and padded % 16 == 0
    assert heads > 0 and 0 < cores <= 64
    d, blocks = 64, padded // 16
    v_row_stride = heads * 3 * d if v_row_stride is None else v_row_stride
    v_head_stride = 3 * d if v_head_stride is None else v_head_stride
    assert v_row_stride % 16 == 0 and v_head_stride % 16 == 0
    assert v_head_stride >= d and v_row_stride >= (heads - 1) * v_head_stride + d
    name = f'vae63_{kind}_s{n}_p{padded}_h{heads}_vr{v_row_stride}_vh{v_head_stride}_c{cores}_{SOURCE}'
    ROOT.mkdir(parents=True, exist_ok=True)
    obj = ROOT / name / (name + '.o')
    with (ROOT / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if obj.exists():
            meta = json.loads(obj.with_suffix('.json').read_text())
            assert meta['blockDim'] == cores
            return str(obj), name + '__kernel0', cores
        platform.set_current_compile_soc_info('Ascend910A')
        t = tik.Tik(disable_debug=True)
        elements = heads * padded * d
        aux = t.Tensor('float16', (16,), name='unused_aux', scope=tik.scope_gm)
        if kind == 'pack':
            q = t.Tensor('float16', (n * heads * d,), name='pack_q', scope=tik.scope_gm)
            k = t.Tensor('float16', (n * heads * d,), name='pack_k', scope=tik.scope_gm)
            vsize = (n - 1) * v_row_stride + (heads - 1) * v_head_stride + d
            v = t.Tensor('float16', (vsize,), name='pack_v', scope=tik.scope_gm)
            y = t.Tensor('float16', (3 * elements,), name='pack_qkv_nz', scope=tik.scope_gm)
            with t.for_range(0, cores, block_num=cores) as core:
                ub = [t.Tensor('float16', (16 * d,), name='pack_ub_%d' % i, scope=tik.scope_ubuf) for i in range(3)]
                with t.for_range(0, (heads * blocks + cores - 1) // cores) as step:
                    task = core + step * cores
                    with t.if_scope(task < heads * blocks):
                        head, block = task // blocks, task % blocks
                        row = block * 16
                        for buf in ub:
                            t.vector_dup(128, buf, 0., 8, 1, 8)
                        with t.if_scope(row + 16 <= n):
                            for src, buf in zip((q, k), ub[:2]):
                                t.data_move(buf, src[row * heads * d + head * d], 0, 16, d // 16, (heads * d - d) // 16, 0)
                            t.data_move(ub[2], v[row * v_row_stride + head * v_head_stride], 0, 16, d // 16, (v_row_stride - d) // 16, 0)
                        if n % 16:
                            with t.else_scope():
                                with t.if_scope(row < n):
                                    for src, buf in zip((q, k), ub[:2]):
                                        t.data_move(buf, src[row * heads * d + head * d], 0, n % 16, d // 16, (heads * d - d) // 16, 0)
                                    t.data_move(ub[2], v[row * v_row_stride + head * v_head_stride], 0, n % 16, d // 16, (v_row_stride - d) // 16, 0)
                        # Match the existing torch FP16 Q*0.125 before cube QK.
                        t.vmuls(128, ub[0], ub[0], .125, 8, 1, 1, 8, 8)
                        for tensor, buf in enumerate(ub):
                            for db in range(d // 16):
                                dst = tensor * elements + head * padded * d + db * padded * 16 + block * 256
                                t.data_move(y[dst], buf[db * 16], 0, 16, 1, d // 16 - 1, 0)
            inputs, outputs = [q, k, v, aux], [y]
        else:
            source = t.Tensor('float16', (elements,), name='unpack_onz', scope=tik.scope_gm)
            aux2 = t.Tensor('float16', (16,), name='unused_aux2', scope=tik.scope_gm)
            aux3 = t.Tensor('float16', (16,), name='unused_aux3', scope=tik.scope_gm)
            y = t.Tensor('float16', (n * heads * d,), name='unpack_bsnd', scope=tik.scope_gm)
            with t.for_range(0, cores, block_num=cores) as core:
                ub = t.Tensor('float16', (d // 16, 16, 16), name='unpack_ub', scope=tik.scope_ubuf)
                with t.for_range(0, (heads * blocks + cores - 1) // cores) as step:
                    task = core + step * cores
                    with t.if_scope(task < heads * blocks):
                        head, block = task // blocks, task % blocks
                        row = block * 16
                        with t.if_scope(row < n):
                            t.data_move(ub, source[head * padded * d + block * 256], 0, d // 16, 16, padded - 16, 0)
                            with t.if_scope(row + 16 <= n):
                                for db in range(d // 16):
                                    t.data_move(y[row * heads * d + head * d + db * 16], ub[db, 0, 0], 0, 16, 1, 0, heads * d // 16 - 1)
                            if n % 16:
                                with t.else_scope():
                                    for db in range(d // 16):
                                        t.data_move(y[row * heads * d + head * d + db * 16], ub[db, 0, 0], 0, n % 16, 1, 0, heads * d // 16 - 1)
            inputs, outputs = [source, aux, aux2, aux3], [y]
        work = Path(tempfile.mkdtemp(prefix='vae63_build_', dir=ROOT)).resolve(strict=True)
        if work.parent != ROOT.resolve() or not work.name.startswith('vae63_build_'):
            raise RuntimeError('Unexpected owned build directory')
        previous = os.getcwd()
        try:
            os.chdir(work)
            t.BuildCCE(kernel_name=name, inputs=inputs, outputs=outputs)
            src = work / 'kernel_meta' / (name + '.o')
            assert src.is_file()
            obj.parent.mkdir(exist_ok=False)
            shutil.copyfile(src.with_suffix('.json'), obj.with_suffix('.json'))
            shutil.copyfile(src, obj)
        finally:
            os.chdir(previous)
            if work.resolve(strict=True) != work or work.parent != ROOT.resolve() or not work.name.startswith('vae63_build_'):
                raise RuntimeError('Unexpected build cleanup target')
            shutil.rmtree(work)
        meta = json.loads(obj.with_suffix('.json').read_text())
        assert meta['blockDim'] == cores
        return str(obj), name + '__kernel0', cores
