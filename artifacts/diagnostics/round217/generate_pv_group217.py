#!/usr/bin/env python3
"""Generate an isolated grouped-K PV candidate from pinned original source."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / 'round215' / 'tik_ops_utils_original.py'
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900'
text = raw.decode()
cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'TikOpsUtils')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'mmad_compute')
signature = text.splitlines()[method.lineno - 1]
body = '''        # R217: retain increasing-K order and the original NZ orientations.
        # Only K grouping changes (16 -> 128); automatic dependency sync stays on.
        M = self.up_align_to_K0(m)
        N = self.up_align_to_K0(n)
        K = self.up_align_to_K0(k)
        C_ub = self.tik_instance.Tensor(precision_type, (N // 16, M, 16), name="C_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            C_l0c = self.tik_instance.Tensor(FP32, (N // 16, M, 16), scope=L0C, name="C_l0c")
            for k_start in range(0, K, 128):
                kg = min(128, K - k_start)
                actual_kg = min(kg, k - k_start)
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    A_l0a = self.tik_instance.Tensor(FP16, (M, kg), scope=L0A, name="A_l0a_group")
                    B_l0b = self.tik_instance.Tensor(FP16, (kg, N), scope=L0B, name="B_l0b_group")
                    # P: K1,M,K0 in L1 -> M1,K1,M0,K0 in L0A.
                    for mi in range(M // 16):
                        self.tik_instance.load2dv1(A_l0a[mi * kg * 16],
                            A_l1[k_start * M + mi * 256], index=0,
                            repeat_times=kg // 16, src_stride=M // 16, sid=0,
                            if_transpose=False)
                    # V keeps N1,K,N0 and the original transposed 16x16 load.
                    for ki in range(kg // 16):
                        self.tik_instance.load2dv1(B_l0b[ki * N * 16],
                            B_l1[(k_start // 16 + ki) * 256], index=0,
                            repeat_times=N // 16, src_stride=K // 16, sid=0,
                            if_transpose=True)
                    self.tik_instance.mmad(C_l0c, A_l0a, B_l0b, m, actual_kg, n,
                        is_bias=0 if k_start == 0 else 1)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1,
                M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
        return self.N1MN0_TO_MN(C_ub) if N1MN0_to_MN else C_ub
'''
lines = text.splitlines(keepends=True)
lines[method.lineno - 1:method.end_lineno] = [signature + '\n' + body]
modified = ''.join(lines)
mc = next(n for n in ast.parse(modified).body if isinstance(n, ast.ClassDef) and n.name == 'TikOpsUtils')
for before, after in zip(cls.body, mc.body):
    if isinstance(before, ast.FunctionDef) and before.name != 'mmad_compute':
        assert ast.dump(before) == ast.dump(after), before.name
(HERE / 'tik_ops_utils_pv_group128.py').write_text(modified)
manifest = {
    'target': 'TikOpsUtils.mmad_compute P@V only', 'k_group': 128,
    'original_sha256': hashlib.sha256(raw).hexdigest(),
    'candidate_sha256': hashlib.sha256(modified.encode()).hexdigest(),
    'original_B_orientation_preserved': 'N1,K,N0 with if_transpose=True load2dv1',
    'automatic_sync_preserved': True,
    'dtype': 'FP16 operands FP32 accumulation/output',
    'rounding_may_change': True, 'nothing_adopted': True,
}
(HERE / 'pv_group_manifest217.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
