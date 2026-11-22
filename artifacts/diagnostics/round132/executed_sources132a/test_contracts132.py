#!/usr/bin/env python3
"""Portable original audio-tail contracts; no model execution."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_sources_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_diagnostic_authorization_not_adoption(self):
        a=json.loads((ROOT/'approval132.json').read_text())
        for key in ('adopt_deterministic_baseline','new_numeric_tolerance','copy_candidate','generation49_or_decoded_media'):self.assertFalse(a[key])
    def test_full50_TP_and_originalchecks(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text();driver=(ROOT/'generate_diagnostics132.py').read_text()
        self.assertIn('VerifiedDiT.forward(model,**inputs)',s);self.assertIn('count==155',s)
        self.assertIn('get_tensor_model_parallel_world_size()==2',s);self.assertIn('get_ulysses_parallel_world_size()==4',s)
        self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",driver)
        self.assertNotIn('omni.generate(',driver)
    def test_nine_full50_and_ten_local_counts_separate(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text();driver=(ROOT/'generate_diagnostics132.py').read_text()
        self.assertIn("[('native_first',False,1),('true_control',True,3),('true_tail_trace',True,2),('true_restored',True,2),('native_final',False,1)]",s)
        self.assertIn('range(10)',s);self.assertIn("'extra_full_model_checks':0",s)
        self.assertIn('checks==11160',driver);self.assertIn('local_calls==80',driver)
    def test_instance_RPC_allrank_str_method(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text();driver=(ROOT/'generate_diagnostics132.py').read_text()
        self.assertIn("name='r132_audio_first_change_diagnostic'",s)
        self.assertIn('def method(self):return diagnostic_rpc(self)',s)
        self.assertNotIn('@classmethod',s);self.assertNotIn('unique_reply_rank=',driver)
        self.assertIn('len(ranks)==8',driver)
    def test_full_head_input_not_model_subsetting(self):
        s=(ROOT/'audio_tail_trace132.py').read_text()
        self.assertIn('x.shape==(41792,5376)',s);self.assertIn('result.shape==(41792,16)',s)
        self.assertIn('result=old(layer,x,bias,*args,**kwargs)',s)
        self.assertNotIn('old(layer,x.index_select',s)
    def test_local_return_before_gather_independent_snapshot(self):
        s=(ROOT/'audio_tail_trace132.py').read_text()
        self.assertIn("current['output']=result.detach().cpu().contiguous().clone()",s)
        self.assertIn('return result  # The observer never substitutes',s)
        self.assertIn('return None  # Preserve original module return',s)
    def test_shared_method_route_by_layer_identity(self):
        s=(ROOT/'audio_tail_trace132.py').read_text()
        self.assertIn('id(layer.quant_method)',s)
        self.assertIn('if layer is not self.audio_layer and layer is not self.video_layer:return old',s)
        self.assertIn('old=old',s);self.assertIn('previous_override',s)
    def test_constructor_transactions(self):
        s=(ROOT/'audio_tail_trace132.py').read_text()
        self.assertIn('try:self._initialize',s);self.assertIn('try:self.close()',s)
        self.assertIn('raise primary from rollback',s)
    def test_full_frozen_word_equivalence(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text()
        self.assertIn("'original_vs_frozen_full_word_proof':frozen_proof",s)
        self.assertIn("compare_fp32(x.detach().cpu().contiguous(),trace.references['x']",s)
        self.assertIn('layer.weight.shape==(16,5376)',s)
        self.assertIn("'physical_NPU_format':'not_measured'",s)
    def test_local_replay_original_apply_no_module(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text().split('def local_replay():',1)[1].split("report['fixed_full_input",1)[0]
        self.assertIn('result=old(layer,x,bias)',s);self.assertNotIn('layer(x)',s)
        self.assertNotIn('all_gather(',s);self.assertNotIn('torch.mm(',s)
    def test_memory_bound_and_chunk_metric(self):
        s=(ROOT/'audio_tail_trace132.py').read_text();metric=(ROOT/'paired_words132.py').read_text()
        self.assertIn('available>=3*expected_bytes*8+2*1024**3',s)
        self.assertIn('free>=2*expected_bytes+64*1024**2',s)
        self.assertIn('chunk_words=262144',metric);self.assertIn('view(np.uint32)',metric)
        self.assertIn('complete_logical_word_range_compared',metric)
    def test_exact_context_and_newroot(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();session=root/'engine_audio132a';session.mkdir()
            source=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_audio13201')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                ns={'__name__':'test132'};exec(compile(source,'context132','exec'),ns);self.assertEqual(ns['SESSION'],session)
    def test_PIDfd_cleanup_deadline_and_originalflag(self):
        for name in ('cohort128.py','private_abort132.py','launch_diagnostics132.py','full_bootstrap127.py','sitecustomize.py'):
            s=(ROOT/name).read_text();self.assertIn('/tmp/h3_audio13201',s);self.assertNotIn('/tmp/h3_full5013101',s)
        s=(ROOT/'private_abort132.py').read_text();self.assertIn('pidfd_send_signal',s);self.assertNotIn('os.kill(',s)
        s=(ROOT/'diagnostic_rpc132.py').read_text();self.assertIn('torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)',s)
    def test_state_FA_collector_safety(self):
        s=(ROOT/'diagnostic_rpc132.py').read_text();self.assertIn('verify_original_R129_master_values',s)
        self.assertIn("fa['actual_ACL_launch_count']==50",s);self.assertIn("r['q_shape'][1]==7",s)
        s=(ROOT/'collect_diagnostics132.py').read_text();self.assertIn("('.json','.jsonl','.txt')",s);self.assertIn('<=32*1024**2',s)


if __name__=='__main__':unittest.main()
