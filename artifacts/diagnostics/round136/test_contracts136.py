#!/usr/bin/env python3
"""Portable source/control contracts for approved full49 quality comparison."""
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
    def test_approval_quality_only(self):
        a=json.loads((ROOT/'approval136.json').read_text());self.assertEqual(a['selected'],'只做完整49步质量对照 (Recommended)')
        for name in ('numeric_baseline_adoption','new_numeric_tolerance','copy_candidate','other_approximation'):self.assertFalse(a[name])
        self.assertTrue(a['combined_deterministic_flag_change_disclosed'])
    def test_original_2_and3_time_rows_no_padding(self):
        s=(ROOT/'fp32_bias_candidate136.py').read_text()
        self.assertIn('x.shape[0] in (2,3)',s);self.assertIn('out_dtype=torch.float32',s)
        self.assertIn('output.add_(bias.float())',s);self.assertEqual(s.count('output.to(torch.float16)'),1)
        self.assertNotIn('bfloat16',s);self.assertNotIn('torch.nn.functional.pad',s);self.assertNotIn('weight.data=',s)
    def test_whole_pipeline_false_true_false(self):
        s=(ROOT/'quality_rpc136.py').read_text()
        self.assertIn("CASES={1:'original_false',2:'candidate_true',3:'restored_false'}",s)
        self.assertIn('torch.use_deterministic_algorithms(index==2,warn_only=False)',s)
        self.assertIn("state['adapter'].candidate.calls==49",s);self.assertIn("{'2':1,'3':48}",s)
    def test_no_extra_generation_or_replay(self):
        s=(ROOT/'generate_quality136.py').read_text()
        self.assertEqual(s.count('omni.generate('),1);self.assertIn('for index,label in LABELS.items()',s)
        self.assertIn('num_inference_steps=49',s);self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",s)
        self.assertIn('actual_full49_requests=3',s)
    def test_original_live_encoder_and_VAE_protected(self):
        s=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn('install_patch_projection_guard123(self)+install_time_projection_guard123(self)',s)
        self.assertIn('count==155',s);self.assertIn('return bundle[',s)
        self.assertIn('assert len(result[name])==50',s)
        self.assertIn('decoder.parameters()',s);self.assertIn('torch.float32',s)
        self.assertIn('original_encode(self,*args,**kwargs)',s);self.assertIn('original_decode(self,video_latent,audio_latent,**kwargs)',s)
    def test_actual_TP_FA_and_candidate_counts(self):
        s=(ROOT/'integration_hooks127.py').read_text();rpc=(ROOT/'quality_rpc136.py').read_text()
        self.assertIn("fa['actual_ACL_launch_count']==50",s);self.assertIn('shape==[1,7,41792,128]',s)
        self.assertIn("candidate_delta==(1 if quality['candidate'] else 0)",s)
        self.assertIn('get_tensor_model_parallel_world_size()==2',rpc)
        self.assertIn('get_ulysses_parallel_world_size()==4',rpc)
    def test_host_only_raw_arrays_preserve_dtype(self):
        s=(ROOT/'generate_quality136.py').read_text()
        self.assertIn("out/'nativeframes136.npy'",s);self.assertIn('np.save(file,frames,allow_pickle=False)',s)
        self.assertIn('np.save(file,audio,allow_pickle=False)',s)
        self.assertNotIn('frames.astype(',s);self.assertNotIn('audio.astype(',s)
    def test_restore_and_failure_monitor(self):
        s=(ROOT/'quality_rpc136.py').read_text();driver=(ROOT/'generate_quality136.py').read_text()
        self.assertIn('restore_local(state)',s);self.assertIn("action=='abort_restore'",s)
        self.assertIn("'phase':'quality_RPC_entry_'",s)
        self.assertIn('model._total_steps127==147',s);self.assertIn('original_master_and_resident_state_unchanged',s)
        self.assertIn("rpc('finish')",driver);self.assertIn("rpc('abort_restore')",driver)
    def test_exact_private_session(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp).resolve();session=base/'engine_quality136a';session.mkdir()
            s=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_quality13601')",f'ROOT=Path({str(base)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                ns={'__name__':'quality_context_test'};exec(compile(s,'context136','exec'),ns);self.assertEqual(ns['SESSION'],session)
        for name in ('launch_quality136.py','cohort128.py','private_abort136.py'):
            s=(ROOT/name).read_text();self.assertIn('/tmp/h3_quality13601',s);self.assertNotIn('/tmp/h3_full5013501',s)
        self.assertIn('pidfd_send_signal',(ROOT/'private_abort136.py').read_text())
    def test_refuse_uncertain_launch(self):
        s=(ROOT/'launch_quality136.py').read_text()
        self.assertIn('not SESSION.exists() and not receipt.exists()',s);self.assertIn('deadline_monotonic=time.monotonic()+5400',s)
        self.assertIn("approval['numeric_baseline_adoption']",s);self.assertIn('original_sampling_fixture_SHA_reverified',s)


if __name__=='__main__':unittest.main()
