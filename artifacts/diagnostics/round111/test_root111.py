"""Stdlib-only source/actual-checkpoint contracts for the REAL H3 conditioning chain."""
from pathlib import Path
import ast
import hashlib
import json
import unittest
ROOT=Path(__file__).resolve().parent


class OriginalConditioning111(unittest.TestCase):
    def test_actual_nativeHalf_matches_original19parameters_and_configuration(self):
        r=json.loads((ROOT/'checkpoint_conditioning111.json').read_text());a,b=r['roots']
        self.assertEqual(a['config']['sha256'],b['config']['sha256']);self.assertEqual(a['index_sha256'],b['index_sha256'])
        pins=json.loads((ROOT/'expected_half_parameters111.json').read_text())
        self.assertTrue(pins['all19Halfvalues_exact_original_conversion']);self.assertEqual(len(pins['parameters']),19)
        self.assertTrue(all(row['source_BFweight_to_Half_exact_bits']for row in pins['parameters'].values()))
        self.assertEqual(b['conditioning_parameters']['token_refiner.blocks.0.attn.qkv_proj.weight']['shape'],[21504,5376])
        self.assertTrue(all(v['dtype']=='F16'for v in b['conditioning_parameters'].values()))
    def test_original_models_and_loader_not_a_manual_NPU_teacher(self):
        s=(ROOT/'construct_original111.py').read_text();ast.parse(s)
        self.assertIn('original.MiniMaxH3TokenRefiner(arch,None',s)
        self.assertIn('original.MiniMaxH3DiTModel.load_weights,model',s)
        self.assertIn('original._reorder_grouped_qkv_to_qkv(value,num_query_groups=56,heads_per_group=1,head_dim=128)',s)
        self.assertIn('block.attn.num_heads==block.attn.num_kv_heads==56',s)
        self.assertIn('assert not block.attn.attention._has_custom_attention',s)
        self.assertIn('params_dtype=torch.float16',s)
        self.assertNotIn('params_dtype=torch.float32',s)
        self.assertIn('max_seqlen=2043,num_requests=1',s)
    def test_original_source_pins_and_real_finalnorm_order(self):
        pins=json.loads((ROOT/'expected_original_sources111.json').read_text())
        for v in pins.values():self.assertEqual(hashlib.sha256(Path(v['local_source']).read_bytes()).hexdigest(),v['sha256'])
        m=(ROOT.parent/'round62/source/minimax_h3_transformer.py').read_text()
        self.assertIn('text_embed = self.token_refiner(',m)
        self.assertIn('return self.final_norm(x)',m)
        self.assertIn('x = x + self.mlp(self.norm2(x))',m)
        self.assertIn('out = self.weight.to(torch.float32) * out',(ROOT/'source/norm.py').read_text())
    def test_actual_capture_sources_and_conditiononly_clock_gates(self):
        r=json.loads((ROOT/'expected_inputs111.json').read_text());self.assertEqual(r['original_refiner_cu_seqlens'],[0,2043,2043])
        self.assertEqual(len(r['inputs']),3);self.assertEqual(r['captured_shape'],[1,2043,5120])
        s=(ROOT/'npu_conditioning111.py').read_text();ast.parse(s)
        self.assertIn("cpu_index['all_original_CPU_baseline_repeats_exact']",s)
        self.assertIn("'not_full_encoder_time':True",s)
        self.assertIn("assert record['repeat_bits_equal']",s)
        self.assertIn('for _ in range(5)',s)
        self.assertLess(s.index('for h in handles:h.remove()'),s.index('for _ in range(5)'))
        self.assertIn("assert helpers.compare(warm.cpu(),first_stages['finalnorm'])['raw_bits_equal']",s)
        self.assertNotIn('torch.bfloat16',s)


if __name__=='__main__':unittest.main()
