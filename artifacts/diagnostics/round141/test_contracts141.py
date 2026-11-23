#!/usr/bin/env python3
"""Contracts for the R141 roofline addendum (measurement only)."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_no_math_change(self):
        s=(ROOT/'roofline_probe141.py').read_text()
        for banned in ('omni.generate(','FinalAdaLN','use_deterministic_algorithms(True','allow_hf32 = True'):
            self.assertNotIn(banned,s)
        self.assertIn('result=original(*args,**kwargs)',s)      # wrapper must call the original
        # The response says where synchronize is allowed: only the standalone H2D
        # microbenchmark may sync; the two in-forward wrappers must not.
        for wrapper in ('def wrapped(','def prefetch('):
            body=s.split(wrapper,1)[1].split('\ndef ',1)[0]
            self.assertNotIn('synchronize',body,'forward wrapper must not synchronize: '+wrapper)
        self.assertEqual(s.count('torch.npu.synchronize()'),2,
                         'synchronize is allowed only in the standalone bandwidth microbenchmark')
    def test_regions_include_gemm_and_block(self):
        h=(ROOT/'benchmark_actual_attention126.py').read_text()
        for region in ("('block', block)","('qkv', block.attn.qkv_proj)","('oproj', block.attn.out_proj)",
                       "('fc1', block.mlp.fc1)","('fc2', block.mlp.fc2)"):
            self.assertIn(region,h)
    def test_tp_reduce_wrapper_target(self):
        s=(ROOT/'roofline_probe141.py').read_text()
        self.assertIn('vllm.model_executor.layers.linear',s)
        self.assertIn('tensor_model_parallel_all_reduce',s)
    def test_result_shape(self):
        b=json.loads((ROOT/'roofline141.json').read_text())
        self.assertTrue(b['measurement_only_no_math_change'])
        self.assertFalse(b['tp_all_reduce']['additive_wall_time'])
        self.assertIn('hook_overhead_note',b)
        names=[r['region'] for r in b['regions']]
        self.assertTrue(any('GEMM subtotal' in n for n in names))
        gemm=next(r for r in b['regions'] if r['region']=='GEMM subtotal')
        self.assertLess(gemm['percent_of_peak'],15)
        self.assertGreater(gemm['device_s'],6.5)
    def test_no_stale_roots(self):
        for name in ('roofline_probe141.py','run_probe141.py','launch_probe141.py'):
            text=(ROOT/name).read_text()
            self.assertNotIn('h3_roofline140',text)
            self.assertNotIn('roofline140',text)

if __name__=='__main__':unittest.main()
