#!/usr/bin/env python3
"""Portable source tests for native repeat diagnostics; no model execution."""
from __future__ import annotations
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_all_sources_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))

    def test_no_generation_or_copy_candidate(self):
        driver=(ROOT/'generate_diagnostics129.py').read_text()
        self.assertNotIn('omni.generate(',driver)
        self.assertNotIn('num_inference_steps=',driver)
        self.assertIn("collective_rpc('r129_native_repeatability'",driver)
        self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",driver)
        diagnostic=(ROOT/'diagnostic_rpc129.py').read_text()
        self.assertNotIn('NativeCopyAdapter128',diagnostic)
        self.assertIn('VerifiedDiT.forward(model,**inputs)',diagnostic)
        self.assertIn('count==155',diagnostic)

    def test_RPC_is_unique_instance_method_and_str(self):
        source=(ROOT/'diagnostic_rpc129.py').read_text()
        self.assertIn("name='r129_native_repeatability'",source)
        self.assertIn('def method(self,mode)',source)
        self.assertIn('assert not hasattr(DiffusionWorker,name)',source)
        self.assertNotIn('@classmethod',source)

    def test_exact_context_import_and_session_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();session=root/'engine_repeat129a';session.mkdir()
            source=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_repeat12901')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                data={'__name__':'test_context129'};exec(compile(source,'runtime_context127.py','exec'),data)
                self.assertEqual(data['SESSION'],session)

    def test_native_contexts_and_cohort_guards(self):
        source=(ROOT/'diagnostic_rpc129.py').read_text()
        for name in ('set_current_vllm_config','set_current_diffusion_config','set_forward_context','get_forward_context'):
            self.assertIn(name,source)
        self.assertIn("cohort.run('full_native_master_and_resident_state_before'",source)
        self.assertIn("cohort.run('publish_completed_before_next_RPC'",source)

    def test_inputs_owned_onhost_not_teacher_features(self):
        source=(ROOT/'diagnostic_rpc129.py').read_text()
        self.assertIn("OLD=Path('/tmp/h3_overlap12801')",source)
        self.assertIn('native_worker_input_step25_rank',source)
        self.assertIn("cohort.run(f'input_before{index}'",source)
        self.assertIn("cohort.run(f'input_after{index}'",source)
        self.assertNotIn('GPU_corrected',source)

    def test_trace_returns_original_outputs(self):
        source=(ROOT/'region_trace129.py').read_text()
        self.assertIn('return None  # nn.Module hook preserves',source)
        self.assertIn('result=old(*args,**kwargs)',source)
        self.assertIn('return result',source)
        self.assertIn('hash_only_not_different_word_count',source)

    def test_scalar_and_inference_tensor_fingerprints(self):
        source=(ROOT/'fingerprints129.py').read_text()
        self.assertIn('cpu.reshape(-1).view(torch.uint8)',source)
        self.assertIn('except RuntimeError:version=None',source)

    def test_master_dtype_metadata_keys_serializable(self):
        source=(ROOT/'diagnostic_rpc129.py').read_text()
        self.assertIn('{str(k):v for k,v in hook.dtype_metadata.items()}',source)

    def test_no_same_shape_mask_assumption_or_kernel_mutation(self):
        source=(ROOT/'diagnostic_rpc129.py').read_text()+(ROOT/'region_trace129.py').read_text()
        for mutation in ('_SCRATCH.clear','_PAD_BUFFERS.clear','allow_hf32=True','install_device_range','ImmutableWeightPool126'):
            self.assertNotIn(mutation,source)

    def test_launcher_strict_auth_and_root(self):
        launch=(ROOT/'launch_diagnostics129.py').read_text();boot=(ROOT/'full_bootstrap127.py').read_text()
        self.assertIn('os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600',launch)
        self.assertIn('not SESSION.exists() and not receipt.exists()',launch)
        self.assertIn("self.name.startswith('DiffusionWorker-')",boot)
        for name in ('launch_diagnostics129.py','generate_diagnostics129.py','full_bootstrap127.py','runtime_context127.py','cohort128.py'):
            text=(ROOT/name).read_text();self.assertIn('/tmp/h3_repeat12901',text);self.assertNotIn('/tmp/h3_runtime12701',text)


if __name__=='__main__':unittest.main()
