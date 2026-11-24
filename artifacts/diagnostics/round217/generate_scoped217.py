#!/usr/bin/env python3
"""Generate one isolated scope-sync experiment; no production files are edited."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'round215' / 'tik_ops_utils_original.py'
EXPECT = '6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900'
raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECT
text = raw.decode()
needle = '''        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)
            # L0C -> ub, fp32 -> fp16 (tensor_mov可做随路转换)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1, M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
'''
assert text.count(needle) == 1
replacement = '''        # R217 conservative control. C_l0c remains scoped exactly as in the original.
        # Reuse the same official low-level generator as Tik.matmul, without the
        # high-level scope marker. Explicit barriers preserve producer/consumer order.
        from tbe.tik.api.cube.matmul import MatMulImpl
        from tbe.tik.api.cube.tik_cube_api import MatmulApiCls
        with self.tik_instance.new_stmt_scope(disable_sync=True):
            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )
            self.tik_instance.pipe_barrier("PIPE_ALL")
            MatMulImpl(True, C_l0c, self.tik_instance).execute(
                MatmulApiCls(A_l1, B_l1, m, k, n, None))
            self.tik_instance.pipe_barrier("PIPE_ALL")
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1, M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
            self.tik_instance.pipe_barrier("PIPE_ALL")
'''
text = text.replace(needle, replacement, 1)
ast.parse(text)
# Assert unchanged bodies outside the targeted routine.
base = ast.parse(raw.decode()); modified = ast.parse(text)
bc = next(n for n in base.body if isinstance(n, ast.ClassDef) and n.name == 'TikOpsUtils')
mc = next(n for n in modified.body if isinstance(n, ast.ClassDef) and n.name == 'TikOpsUtils')
for a, b in zip(bc.body, mc.body):
    if isinstance(a, ast.FunctionDef) and a.name != 'matmul_compute':
        assert ast.dump(a) == ast.dump(b), a.name
out = HERE / 'tik_ops_utils_scoped_all.py'
out.write_text(text)
manifest = {'original_sha256': EXPECT, 'candidate_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'target': 'TikOpsUtils.matmul_compute only', 'scope_allocation_preserved': True,
            'explicit_barriers': 'PIPE_ALL before matmul, before transfer, at end',
            'nothing_adopted': True}
(HERE / 'source_manifest217.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
