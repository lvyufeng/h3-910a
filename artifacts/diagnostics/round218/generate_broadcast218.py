#!/usr/bin/env python3
"""Replace only FP16 vector-to-row broadcast with an exact UB transpose-scatter."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
original=HERE.parent/'round215'/'tik_ops_utils_original.py'
raw=original.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900'
text=raw.decode()
cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='TikOpsUtils')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='broadcast_vec_from_M_to_MN0')
replacement='''    def broadcast_vec_from_M_to_MN0(self, vec_ub):
        """R218 exact bit-copy broadcast: transpose sixteen identical source rows."""
        M = vec_ub.shape[0]
        assert M % 16 == 0
        dst_ub = self.tik_instance.Tensor(FP16, (M, 16), name="dst_ub", scope=UB)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # Each 16x16 input tile consists of 16 aliases of the same 16-value
            # vector. Transposing it yields 16 rows of repeated FP16 bit patterns.
            dst_list = [dst_ub[16 * i] for i in range(16)]
            src_list = [vec_ub for _ in range(16)]
            repeats = M // 16
            self.tik_instance.vec_trans_scatter(
                False, False, dst_list, src_list, repeats,
                16 if repeats > 1 else 0, 1 if repeats > 1 else 0)
        return dst_ub
'''
lines=text.splitlines(keepends=True)
lines[method.lineno-1:method.end_lineno]=[replacement]
modified=''.join(lines)
mc=next(n for n in ast.parse(modified).body if isinstance(n,ast.ClassDef) and n.name=='TikOpsUtils')
for a,b in zip(cls.body,mc.body):
    if isinstance(a,ast.FunctionDef) and a.name!=method.name:assert ast.dump(a)==ast.dump(b),a.name
(HERE/'tik_ops_utils_broadcast218.py').write_text(modified)
manifest={'target':'broadcast_vec_from_M_to_MN0 only','candidate_sha256':hashlib.sha256(modified.encode()).hexdigest(),
          'original_sha256':hashlib.sha256(raw).hexdigest(),
          'bit_copy_only_no_arithmetic':True,'automatic_sync_preserved':True,'nothing_adopted':True}
(HERE/'broadcast_manifest218.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
