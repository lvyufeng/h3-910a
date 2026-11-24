#!/usr/bin/env python3
"""Generate grouped-K double-buffer PV candidate (separate from other overlays)."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
src=HERE.parent/'round215'/'tik_ops_utils_original.py'
raw=src.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900'
text=raw.decode()
cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='TikOpsUtils')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mmad_compute')
signature=text.splitlines()[method.lineno-1]
body='''        # R217 grouped 128-wide K loads using the original thread_num=2 pattern.
        # A/B NZ orientations, output FP32 and automatic dependency sync are kept.
        M = self.up_align_to_K0(m)
        N = self.up_align_to_K0(n)
        K = self.up_align_to_K0(k)
        C_ub = self.tik_instance.Tensor(precision_type, (N // 16, M, 16), name="C_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            C_l0c = self.tik_instance.Tensor(FP32, (N // 16, M, 16), scope=L0C, name="C_l0c")
            def issue_group(k_start, kg, actual_kg, first):
                A_l0a = self.tik_instance.Tensor(FP16, (M, kg), scope=L0A, name="A_l0a_g2")
                B_l0b = self.tik_instance.Tensor(FP16, (kg, N), scope=L0B, name="B_l0b_g2")
                for mi in range(M // 16):
                    self.tik_instance.load2dv1(A_l0a[mi * kg * 16],
                        A_l1[k_start * M + mi * 256], index=0,
                        repeat_times=kg // 16, src_stride=M // 16, sid=0,
                        if_transpose=False)
                for ki in range(kg // 16):
                    self.tik_instance.load2dv1(B_l0b[ki * N * 16],
                        B_l1[(k_start // 16 + ki) * 256], index=0,
                        repeat_times=N // 16, src_stride=K // 16, sid=0,
                        if_transpose=True)
                if isinstance(first, bool):
                    self.tik_instance.mmad(C_l0c, A_l0a, B_l0b, m, actual_kg, n,
                                          is_bias=0 if first else 1)
                else:
                    with self.tik_instance.if_scope(first):
                        self.tik_instance.mmad(C_l0c, A_l0a, B_l0b, m, actual_kg, n, is_bias=0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.mmad(C_l0c, A_l0a, B_l0b, m, actual_kg, n, is_bias=1)
            full = K // 128
            pairs = full // 2
            if pairs:
                with self.tik_instance.for_range(0, pairs) as pair:
                    with self.tik_instance.for_range(0, 2, thread_num=2) as thread:
                        first = tik.all(pair == 0, thread == 0)
                        issue_group((pair * 2 + thread) * 128, 128, 128, first)
            if full % 2:
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    issue_group((full - 1) * 128, 128,
                                min(128, k - (full - 1) * 128), full == 1)
            tail = K - full * 128
            if tail:
                with self.tik_instance.new_stmt_scope(disable_sync=False):
                    issue_group(full * 128, tail, k - full * 128, full == 0)
            self.tik_instance.tensor_mov(C_ub, C_l0c, "m", 1,
                M * N * DTYPE_SIZE[FP32] // 1024, 0, 0)
        return self.N1MN0_TO_MN(C_ub) if N1MN0_to_MN else C_ub
'''
lines=text.splitlines(keepends=True)
lines[method.lineno-1:method.end_lineno]=[signature+'\n'+body]
modified=''.join(lines)
mc=next(n for n in ast.parse(modified).body if isinstance(n,ast.ClassDef) and n.name=='TikOpsUtils')
for a,b in zip(cls.body,mc.body):
    if isinstance(a,ast.FunctionDef) and a.name!='mmad_compute':assert ast.dump(a)==ast.dump(b),a.name
(HERE/'tik_ops_utils_pv_group128_double.py').write_text(modified)
manifest={'target':'P@V mmol utility','k_group':128,'thread_num':2,
          'original_sha256':hashlib.sha256(raw).hexdigest(),
          'candidate_sha256':hashlib.sha256(modified.encode()).hexdigest(),
          'automatic_sync_preserved':True,'rounding_may_change':True,'nothing_adopted':True}
(HERE/'pv_group_double_manifest217.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
