#!/usr/bin/env python3
"""Portable control tests for approved full50 candidate diagnostics, no computation."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_approval_not_adoption(self):
        a=json.loads((ROOT/'approval135.json').read_text());self.assertEqual(a['selected'],'只做全50层候选诊断 (Recommended)')
        for key in ('baseline_adoption','new_numeric_tolerance','generation49_or_decoded_media','copy_candidate'):self.assertFalse(a[key])
    def test_unique_target_routing_and_exact_restore(self):
        s=(ROOT/'candidate_adapter135.py').read_text()
        self.assertIn('model.final_layer.adaln_proj.linear',s);self.assertIn('if layer is not self.target:',s)
        self.assertIn('return self.old(layer,x,bias,*args,**kwargs)',s)
        self.assertIn('return self.candidate.apply(layer,x,bias)',s);self.assertIn("delattr(self.method,'apply')",s)
    def test_original_half_candidate_same_as_local(self):
        s=(ROOT/'fp32_bias_candidate134.py').read_text()
        self.assertIn('torch.mm(x,layer.weight.t(),out_dtype=torch.float32)',s)
        self.assertIn('output.add_(bias.float())',s);self.assertEqual(s.count('output.to(torch.float16)'),1)
        self.assertNotIn('weight.data=',s)
    def test_full50_strict_TP_and_nosampler(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text();driver=(ROOT/'generate_diagnostics135.py').read_text()
        self.assertIn('VerifiedDiT.forward(model,**inputs)',s);self.assertIn('count==155',s)
        self.assertIn('len(model.blocks)==50',s);self.assertIn('get_tensor_model_parallel_world_size()==2',s)
        self.assertIn('get_ulysses_parallel_world_size()==4',s)
        self.assertNotIn('omni.generate(',driver);self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",driver)
    def test_threecases_same_true_flag(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text()
        self.assertIn("for case in ('original','candidate','restored')",s)
        self.assertIn('torch.use_deterministic_algorithms(True,warn_only=False)',s)
        self.assertIn('range(3)',s);self.assertIn("if case=='candidate':adapter.install()",s)
        self.assertIn('torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)',s)
    def test_actual_one_candidate_call_per_forward(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text();driver=(ROOT/'generate_diagnostics135.py').read_text()
        self.assertIn("adapter.candidate.calls-candidate_before==(1 if case=='candidate' else 0)",s)
        self.assertIn('adapter.candidate.calls==3',s);self.assertIn('checks==11160',driver)
        self.assertIn("sum(r['candidate_actual_local_calls'] for r in ranks)==24",driver)
    def test_state_and_numeric_restore_separate(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text()
        for key in ('output_vs_same_case_first','output_vs_original_first','output_vs_original_last','candidate_method_physically_restored','private_flags_physically_restored','model_before_state','model_after_state'):
            self.assertIn(key,s)
        self.assertIn('verify_original_R129_master_SHA_reference',s)
        self.assertIn("attempt('candidate_adapter_restore',adapter.restore)",s)
        self.assertIn("attempt('initial_false_warn_flag_restore'",s)
        self.assertIn('nonlocal time_handle,activity',s)
    def test_FA_stream_and_realhead_shape(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text();self.assertIn("fa['actual_ACL_launch_count']==50",s)
        self.assertIn('len(shape)==4 and shape[1]==7',s);self.assertIn('Keep original shared-copy baseline',s)
    def test_RPC_allranks_str_and_unique_instance(self):
        s=(ROOT/'diagnostic_rpc135.py').read_text();driver=(ROOT/'generate_diagnostics135.py').read_text()
        self.assertIn("name='r135_full50_final_AdaLN_candidate_diagnostic'",s)
        self.assertNotIn('@classmethod',s);self.assertNotIn('unique_reply_rank=',driver);self.assertIn('len(ranks)==8',driver)
    def test_exactroot_context_and_cleanup(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();session=root/'engine_candidate135a';session.mkdir()
            s=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_full5013501')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                ns={'__name__':'contexttest135'};exec(compile(s,'context135','exec'),ns);self.assertEqual(ns['SESSION'],session)
        for name in ('launch_diagnostics135.py','private_abort135.py','cohort128.py'):
            s=(ROOT/name).read_text();self.assertIn('/tmp/h3_full5013501',s);self.assertNotIn('/tmp/h3_full5013101',s)
        self.assertIn('pidfd_send_signal',(ROOT/'private_abort135.py').read_text())
    def test_collector_only_scalar_text(self):
        s=(ROOT/'collect_diagnostics135.py').read_text();self.assertIn("('.json','.jsonl','.txt')",s)
        self.assertIn('<=32*1024**2',s);self.assertIn('full50_rank*135.json',s)


if __name__=='__main__':unittest.main()
