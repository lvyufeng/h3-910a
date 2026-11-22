#!/usr/bin/env python3
"""Portable contracts for authorized full50 diagnostics, no model execution."""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_all_sources_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))

    def test_explicit_approval_not_adoption(self):
        approval=json.loads((ROOT/'approval131.json').read_text())
        self.assertEqual(approval['selected'],'只做全50层诊断 (Recommended)')
        for key in ('model_candidate_adoption_authorized','new_numeric_tolerance_authorized','copy_candidate_authorized','generation49_authorized_for_this_experiment'):
            self.assertFalse(approval[key])

    def test_original_full50_TP_and_strict_checks(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn('VerifiedDiT.forward(model,**inputs)',source)
        self.assertIn('count==155',source);self.assertIn('len(model.blocks)==50',source)
        self.assertIn('get_tensor_model_parallel_world_size()==2',source)
        self.assertIn('model.blocks[0].attn.num_heads==28',source)
        driver=(ROOT/'generate_diagnostics131.py').read_text()
        self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",driver)
        self.assertIn("'ulysses_degree':4,'ring_degree':1",driver)
        self.assertNotIn('omni.generate(',driver)

    def test_unique_RPC_instance_str_all_ranks(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text();driver=(ROOT/'generate_diagnostics131.py').read_text()
        self.assertIn("name='r131_full50_deterministic_diagnostic'",source)
        self.assertIn('def method(self):return diagnostic_rpc(self)',source)
        self.assertNotIn('@classmethod',source)
        self.assertIn("collective_rpc('r131_full50_deterministic_diagnostic'",driver)
        self.assertNotIn('unique_reply_rank=',driver)
        self.assertIn('len(reports)==8',driver)

    def test_native_deterministic_restored_no_tolerance(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn("[('native',False),('deterministic',True),('restored',False)]",source)
        self.assertIn('range(3)',source)
        self.assertIn('torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)',source)
        self.assertIn("'private_flags_physically_restored'",source)
        self.assertIn("'new_precision_policy_adopted':False",source)
        self.assertNotIn('error_tolerance',source)

    def test_time_capture_preserves_result_and_stream_order(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn("time_holder['time']=output.detach().clone()",source)
        self.assertIn('return None  # Preserve original module return',source)
        capture=source.split('def capture_time(',1)[1].split('holder={}',1)[0]
        self.assertNotIn('.cpu()',capture)
        self.assertNotIn('synchronize()',capture)

    def test_source_inputs_and_weights_held_on_host(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn("OLD=Path('/tmp/h3_overlap12801')",source)
        self.assertIn('native_worker_input_step25_rank',source)
        self.assertIn('hashlib.file_digest',source)
        self.assertIn('native_CPU_masters_words_metadata_unchanged',source)
        self.assertIn('materialized_parameters_buffers_unchanged',source)

    def test_numeric_scopes_not_confused_with_restoration(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        for key in ('output_vs_same_case_first','output_vs_native_first','output_vs_native_last','time_vs_same_case_first','time_vs_native_first'):
            self.assertIn(key,source)
        self.assertIn('private_flags_physically_restored',source)
        self.assertNotIn("report['restored']=",source)

    def test_FA_hits_original_stream_no_candidate(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn("fa['actual_ACL_launch_count']==50",source)
        self.assertIn("fa['FA_wrapper_error_count']==0",source)
        self.assertIn('Keep original shared-copy baseline',source)
        self.assertNotIn('NativeCopyAdapter128',source)

    def test_memory_timer_and_case_scope(self):
        source=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn('torch.npu.reset_peak_memory_stats()',source)
        self.assertIn('local_full50_including_original_audit_and_output_D2H_s',source)
        self.assertIn('Entire private process during each case',source)

    def test_exact_private_root_session_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();session=root/'engine_deterministic131a';session.mkdir()
            source=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_full5013101')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                namespace={'__name__':'test_context131'};exec(compile(source,'runtime_context127.py','exec'),namespace)
                self.assertEqual(namespace['SESSION'],session)

    def test_root_guards_and_detached_cleanup(self):
        for name in ('launch_diagnostics131.py','full_bootstrap127.py','sitecustomize.py','runtime_context127.py','cohort128.py','private_abort131.py'):
            source=(ROOT/name).read_text()
            self.assertIn('/tmp/h3_full5013101',source);self.assertNotIn('/tmp/h3_repeat12902',source)
        source=(ROOT/'launch_diagnostics131.py').read_text()+(ROOT/'cohort128.py').read_text()
        self.assertIn('abort_private131()',source)
        self.assertNotIn('os.killpg(process.pid,signal.SIGTERM)',source)
        abort=(ROOT/'private_abort131.py').read_text()
        self.assertIn("'start_ticks'",abort);self.assertIn('stat().st_uid==os.getuid()',abort)

    def test_signals_bound_to_PIDfds_and_post_exit_reconciliation(self):
        source=(ROOT/'private_abort131.py').read_text();launcher=(ROOT/'launch_diagnostics131.py').read_text()
        self.assertIn('signal.pidfd_send_signal(fd,signal.SIGTERM)',source)
        self.assertIn('signal.pidfd_send_signal(fd,signal.SIGKILL)',source)
        self.assertNotIn('os.kill(',source)
        self.assertIn('with _ABORT_LOCK',source)
        self.assertIn('post_parent_exit_private_cleanup',launcher)
        self.assertIn('all_private_children_exited',launcher)

    def test_total_deadline_and_real_SP_evidence(self):
        source=(ROOT/'cohort128.py').read_text();launcher=(ROOT/'launch_diagnostics131.py').read_text()
        self.assertIn('deadline_monotonic-time.monotonic()',source)
        self.assertIn("abort('total_90minute_compute_budget_exceeded')",source)
        self.assertIn('select.select([parent_fd]',launcher)
        diagnostic=(ROOT/'diagnostic_rpc131.py').read_text()
        self.assertIn('get_ulysses_parallel_world_size()==4',diagnostic)
        self.assertIn('get_sequence_parallel_world_size()==4',diagnostic)
        self.assertIn('len(shape)==4 and shape[1]==7',diagnostic)
        self.assertIn("if case!='native' else None",diagnostic)

    def test_collector_excludes_model_and_activation_archives(self):
        source=(ROOT/'collect_diagnostics131.py').read_text()
        self.assertIn("('.json','.jsonl','.txt')",source)
        self.assertIn('<=32*1024**2',source)
        self.assertIn('assert not args.output_dir.exists()',source)
        self.assertIn('full50_rank*131.json',source)


if __name__=='__main__':unittest.main()
