#!/usr/bin/env python3
"""Private exact logical-bound trimming for padding only; storage NZ strides unchanged."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
text=(HERE/'fa218.py').read_text()
old='''    key = _kernel_cache_key(b, h, n, d, br, bc, core_budget, key_seq_limit)
'''
new='''    key = _kernel_cache_key(b, h, n, d, br, bc, core_budget, key_seq_limit)
    source_q_limit = int(os.environ["H3_FA_TRIM_Q_LIMIT"]) if os.environ.get("H3_FA_TRIM_Q_LIMIT") else None
    if source_q_limit is not None:
        assert 0 < source_q_limit <= n and key_seq_limit is not None
        key = "trim218_q%d_" % source_q_limit + key
'''
assert text.count(old)==1;text=text.replace(old,new,1)
old='''                    instance.compute_process()
                    # R217: record actual geometry and mode, not the requested cap
'''
new='''                    if source_q_limit is not None:
                        # Retain GM Nq/N/NZ shapes and strides; only schedule useful
                        # Q tiles and KV tiles containing at least one valid key.
                        # Last tiles retain original full tile width and exact masks.
                        import types
                        original_init = instance.init
                        def trim_init(self):
                            original_init()
                            assert not self.has_attn_mask and not self.has_drop_mask and not self.has_alibi_mask
                            assert self.Nq % self.Br == 0 and self.N % self.Bc == 0
                            self.Tr = (source_q_limit + self.Br - 1) // self.Br
                            self.Tc = (key_seq_limit + self.Bc - 1) // self.Bc
                            self.last_Br = self.Br
                            self.last_Bc = self.Bc
                        instance.init = types.MethodType(trim_init, instance)
                    instance.compute_process()
                    # R217: record actual geometry and mode, not the requested cap
'''
assert text.count(old)==1;text=text.replace(old,new,1)
text=text.replace('"key_seq_limit": key_seq_limit,','"key_seq_limit": key_seq_limit, "source_q_limit": source_q_limit,')
ast.parse(text)
(HERE/'fa218trim.py').write_text(text)
probe=(HERE/'fa_probe218.py').read_text().replace("Path('/tmp/h3_broadcast218')","Path('/tmp/h3_trim218')").replace('import fa218 as fa','import fa218trim as fa')
ast.parse(probe)
(HERE/'trim_probe218.py').write_text(probe)
print('trim storage layout untouched; logical Tr/Tc only; output kept for source Q prefix')
