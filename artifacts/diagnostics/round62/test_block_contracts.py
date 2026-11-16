#!/usr/bin/env python3
"""Regression gates for actual DiT block replay scope and reference identity."""
import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class BlockContracts(unittest.TestCase):
    def test_actual_fixtures_have_all_ranks_layers_and_global_keys(self):
        rows=json.loads((ROOT/'capture_metadata'/'capture_metadata.json').read_text())
        self.assertEqual(len(rows),24)
        self.assertEqual({(r['rank'],r['layer']) for r in rows},{(r,l) for r in range(8) for l in (0,24,49)})
        for r in rows:
            self.assertEqual(r['shape'],[5224,5376])
            self.assertEqual(r['kwargs']['packed_total'],41792)
            self.assertEqual(r['kwargs']['max_seqlen'],41769)
            self.assertEqual(r['kwargs']['rope_table'],[5224,96])
            self.assertLess(r['tensor_bytes'],512*1024*1024)

    def test_ctor_and_forward_use_real_sp_context(self):
        text=(ROOT/'block_replay_tp8.py').read_text();tree=ast.parse(text)
        contexts=[n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='set_forward_context' for i in n.items)]
        self.assertGreaterEqual(len(contexts),2)
        self.assertIn("parallel_strategy.name=='ulysses'",text)
        self.assertIn('tensor_parallel_size=1',text)
        self.assertIn('ulysses_degree=size',text)
        self.assertIn('loaded==set(dict(owner.named_parameters())) and len(loaded)==10',text)
        self.assertIn("owner._diffusers_weights=False",text)

    def test_all_actual_native_and_split_outputs_match(self):
        rows=[json.loads(p.read_text()) for p in (ROOT/'block_replay').glob('block_replay02_rank*.json')]
        self.assertEqual(len(rows),8)
        for r in rows:
            self.assertTrue(r['completed']);self.assertEqual(len(r['layers']),3)
            for layer in r['layers']:
                for mode in ('baseline','point_gate','lossless_split'):
                    c=layer['cases'][mode]
                    self.assertTrue(c['vs_capture']['raw_bits_equal'])
                    self.assertTrue(c['vs_baseline']['raw_bits_equal'])
                    for name in ('first_second','second_third','third_fourth'):self.assertTrue(c[name]['raw_bits_equal'])

    def test_faster_norm_is_explicitly_nonlossless(self):
        summary=json.loads((ROOT/'summary.json').read_text())
        self.assertEqual(summary['exact_outputs_per_accepted_mode'],674021376)
        self.assertTrue(summary['norm_fusion_lossless_rejected'])
        self.assertFalse(summary['production_fusion_installed'])
        self.assertFalse(summary['goal_completed'])
        self.assertFalse(summary['official_matched_quality_benchmark_completed'])
        for layer in summary['per_layer']:
            self.assertFalse(layer['cases']['norm_fused_exploratory']['all_baselines_bits_equal'])
            self.assertEqual(len(layer['cases']['baseline']['max_rank_warm_groups_ms']),3)
        self.assertIn('exclude',summary['scope'])

    def test_executed_final_sources_and_restore_receipts_match(self):
        receipt=json.loads((ROOT/'final_source_verification.json').read_text())
        self.assertTrue(receipt['all_production_unchanged'])
        self.assertFalse(receipt['production_fusion_installed'])
        for root,name in [('round61','adaln_vector61.py'),('round61','point_vector61.py'),('round61','vec61_launcher.c'),('round61','point_guard_validation.py'),('round62','block_replay_tp8.py')]:
            local=ROOT.parent/root/name
            remote='/tmp/h3_round61_adaln01/'+name if root=='round61' else '/tmp/h3_round62_blocks01/'+name
            self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),receipt['diagnostic_sources'][remote])
        restored=json.loads((ROOT/'capture_restoration.json').read_text())
        self.assertTrue(restored['restored'])
        self.assertEqual(restored['sha256'],hashlib.sha256((ROOT/'source'/'minimax_h3_transformer.py').read_bytes()).hexdigest())

    def test_failed_no_parallel_replay_is_retained_not_accepted(self):
        row=json.loads((ROOT/'attempt01'/'block_replay_rank0.json').read_text())
        self.assertEqual(row['layers'],[])
        self.assertIn('41792',row['failure']['message'])
        self.assertIn('5224',row['failure']['message'])
        self.assertNotIn('completed',row)


if __name__=='__main__':unittest.main(verbosity=2)
