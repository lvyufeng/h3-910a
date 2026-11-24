#!/usr/bin/env python3
"""R217 independent PV matmul candidate; unlike scoped-sync experiment, automatic sync stays on."""
from pathlib import Path
import ast, hashlib, json
HERE=Path(__file__).resolve().parent
src=HERE.parent/'round215'/'flash_attention_fwd_original.py'
raw=src.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='a21c52eaa65071e6d496449bf663571bc543b1cdd818fe8950b9764e1680ffdd'
text=raw.decode()
old='''            Oij_ub_zN = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, q_blk_h, kv_blk_h, self.actual_d,
                                                        N1MN0_to_MN=False, precision_type=FP32)'''
new='''            # R217 PV candidate: same L1 operands, same FP32 result and masks.
            # Let the official tiler use wider K chunks instead of hand-issuing
            # 16-wide K accumulation. Rounding may differ; measurement only.
            Oij_ub_zN = self.tik_ops_utils.matmul_compute(Pij_l1_zN, Vj_l1_zN, q_blk_h, kv_blk_h, self.actual_d,
                                                        N1MN0_to_MN=False, precision_type=FP32)'''
assert text.count(old)==1
text=text.replace(old,new,1)
ast.parse(text)
a=ast.parse(raw.decode());b=ast.parse(text)
ac=next(x for x in a.body if isinstance(x,ast.ClassDef));bc=next(x for x in b.body if isinstance(x,ast.ClassDef))
for before,after in zip(ac.body,bc.body):
    if isinstance(before,ast.FunctionDef) and before.name!='output_compute':
        assert ast.dump(before)==ast.dump(after),before.name
out=HERE/'flash_attention_fwd_pv.py'
out.write_text(text)
manifest={'target':'FlashAttentionFwd.output_compute P@V only','original_sha256':hashlib.sha256(raw).hexdigest(),
          'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),'automatic_sync_preserved':True,
          'attention_mask_and_key_limit_preserved':True,'tile_caps_unchanged':True,
          'accumulation_output_dtype':'FP32','rounding_may_change':True,'nothing_adopted':True}
(HERE/'pv_manifest217.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
