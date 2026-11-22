#!/usr/bin/env python3
"""Portable source controls for the original small final-AdaLN probe."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    def test_original_modules_and_time_reference(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('MiniMaxH3TimeEmbedder',s);self.assertIn('MiniMaxH3AdalnProj',s)
        self.assertIn('install_time_projection_guard123(holder)',s)
        self.assertIn('actual_formal_t_emb_SHA_matched',s);self.assertIn('assert actual==ref',s)
        self.assertNotIn('MiniMaxH3DiTModel',s);self.assertNotIn('.generate(',s)
    def test_half_word_count_and_original_linear_call(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('view(np.uint16)',s);self.assertIn('complete_logical_words_compared',s)
        self.assertIn('output=old(actual_layer,x,bias)',s)
        self.assertIn("lambda:old(layer,fixed['x'],fixed['bias'])",s)
        self.assertNotIn('torch.mm(',s);self.assertNotIn('torch.nn.functional.linear(',s)
    def test_original_full_shapes(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('layer.weight.shape==(5376,2688)',s)
        self.assertIn('x.shape==(3,2688)',s);self.assertIn('bias.dtype==torch.float16',s)
        self.assertIn('expand_ratio=2,modality_num=1',s)
    def test_native_config_and_dispatch_proof_scope(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('create_diffusion_vllm_config',s);self.assertIn('with set_current_vllm_config(native_vllm)',s)
        self.assertIn('same_python_default_dispatch_as_R132_audio_head',s)
        self.assertIn('not an independently recorded R132 final-AdaLN runtime config',s)
        self.assertIn('source_input_SHA_verified',s)
    def test_allrank_gates_for_module_and_no_earlyexit(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('construct_original_small_components_all8',s)
        self.assertIn('verify_formal_t_emb_all8_before_Module',s)
        self.assertIn('original_Module_with_TP_gather_',s)
        self.assertIn('original_local_apply_no_TP_',s)
        self.assertIn('publish_fatal128',s)
        self.assertIn('range(32)',s);self.assertIn('range(128)',s)
    def test_physical_restore_and_no_policy(self):
        s=(ROOT/'component133.py').read_text()
        self.assertIn('torch.use_deterministic_algorithms(initial,warn_only=warn)',s)
        self.assertIn('wrapper_installed',s);self.assertIn("delattr(method,'apply')",s)
        self.assertIn("'new_arithmetic_or_flag_baseline_adopted':False",s)
    def test_bound_PIDfd_cleanup_and_deadline(self):
        s=(ROOT/'launch_component133.py').read_text();cleanup=(ROOT/'private_abort133.py').read_text()
        self.assertIn('deadline=time.monotonic()+1200',s);self.assertIn('monitor_private_failure(',s)
        self.assertIn('select.select([fd]',s);self.assertIn('all_private_processes_exited',s)
        self.assertIn('pidfd_send_signal',cleanup);self.assertNotIn('os.kill(',cleanup)
        self.assertIn('H3_R133_PRIVATE_ROOT=/tmp/h3_adaln13301',cleanup)


if __name__=='__main__':unittest.main()
