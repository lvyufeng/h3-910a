#!/usr/bin/env python3
"""Structural causal mask/cache and diagnostic scope regressions."""
import ast
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parent
class CausalContracts(unittest.TestCase):
 def test_triangular_packed_mask_rows(self):
  n=128
  for row in range(n):
   masked=[]
   block,valid=divmod(row+1,16)
   if valid:
    mask=((1<<16)-1)^((1<<valid)-1)
    masked.extend(block*16+i for i in range(16) if mask&(1<<i));block+=1
   for b in range(block,n//16):masked.extend(range(b*16,(b+1)*16))
   self.assertEqual(masked,list(range(row+1,n)))
 def test_tbe_causal_cache_and_metadata_isolated(self):
  text=(ROOT/'candidate_causal_fa910a.py').read_text()
  self.assertIn('"_causal_v4" if causal else key',text)
  self.assertIn('"fa910a_kernel_version"] = 4 if causal else 3',text)
  self.assertIn('bc = br if causal else',text)
  self.assertIn('fill_tail(self, scores, kv_index, 0.0)',text)
 def test_fp32_math_has_no_low_precision_workspace(self):
  text=(ROOT/'h3_text_causal57.py').read_text()
  self.assertIn("with torch.autocast(device_type=q.device.type,enabled=False):",text)
  self.assertIn('qf=q.float().contiguous()',text)
  self.assertIn('scores=scores.masked_fill(',text)
  self.assertNotIn('.half()',text)
  self.assertIn('return torch.matmul(probabilities,vf).to(q.dtype)',text)
 def test_default_off_and_bounded_shapes(self):
  text=(ROOT/'h3_text_causal57.py').read_text()
  self.assertIn("get('VLLM_OMNI_910A_TEXT_CAUSAL_FP32','0')",text)
  self.assertIn('_MAX_SEQUENCE=4096',text)
  self.assertIn('_MAX_QUERY_HEADS=16',text)
  self.assertIn('_MAX_MASKS=4',text)
 def test_diagnostic_output_hooks_do_not_modify_math(self):
  tree=ast.parse((ROOT/'text_attention_probe.py').read_text())
  hooks=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in ('pre','post')]
  self.assertEqual(len(hooks),2)
  for n in hooks:
   self.assertTrue(all(isinstance(v.value,ast.Constant) and v.value.value is None for v in ast.walk(n) if isinstance(v,ast.Return)))
 def test_fast_causal_partition_counts_only_prefix_work(self):
  text=(ROOT/'candidate_causal_fast_fa910a.py').read_text()
  self.assertIn('instance.next_block_num = 0',text)
  self.assertIn('return tr_idx + 1',text)
  self.assertIn('_causal_v5',text)
  self.assertIn('5 if causal else 3',text)
  self.assertIn('fill_tail(self, scores, kv_index, 0.0)',text)

if __name__=='__main__':unittest.main(verbosity=2)
