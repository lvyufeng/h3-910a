#!/usr/bin/env python3
"""Create a private wrapper that records actual build geometry, mask mode and source hashes."""
from pathlib import Path
import ast

HERE=Path(__file__).resolve().parent
source=HERE.parent/'round215'/'fa910a_sweep.py'
text=source.read_text()
needle='''                    instance.compute_process()
                    src = os.path.join(work, "kernel_meta", "%s.o" % key)
'''
replacement='''                    instance.compute_process()
                    # R217: record actual geometry and mode, not the requested cap
                    # or a FLOP count inferred from a tiling strategy's name.
                    import inspect
                    from flash_attention.ops.flash_attention.tik_ops_utils import TikOpsUtils
                    build_record = {
                        "B_heads": int(instance.B), "head_num": int(instance.head_num),
                        "Nq": int(instance.Nq), "Nk": int(instance.N), "d": int(instance.actual_d),
                        "Br": int(instance.Br), "Bc": int(instance.Bc),
                        "Tr": int(instance.Tr), "Tc": int(instance.Tc),
                        "core_num": int(instance.core_num),
                        "prev_block_num": int(instance.prev_block_num),
                        "next_block_num": int(instance.next_block_num),
                        "has_attn_mask": bool(instance.has_attn_mask),
                        "only_causal": bool(getattr(instance, "only_causal", False)),
                        "key_seq_limit": key_seq_limit,
                    }
                    for kind, cls in [("forward", FlashAttentionFwd), ("utils", TikOpsUtils)]:
                        path = inspect.getfile(cls)
                        with open(path,"rb") as source_file:
                            build_record[kind+"_sha256"] = hashlib.sha256(source_file.read()).hexdigest()
                        build_record[kind+"_path"] = path
                    src = os.path.join(work, "kernel_meta", "%s.o" % key)
                    with open(src.replace(".o", ".json")) as meta_file:
                        r217_meta=json.load(meta_file)
                    r217_meta["r217_actual_instance"] = build_record
                    with open(src.replace(".o", ".json"), "w") as meta_file:
                        json.dump(r217_meta, meta_file)
                    print("[R217_BUILD] " + json.dumps(build_record), flush=True)
'''
assert text.count(needle)==1
text=text.replace(needle,replacement,1)
ast.parse(text)
(HERE/'fa217.py').write_text(text)
print('generated fa217.py',len(text))
