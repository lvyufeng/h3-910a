#!/usr/bin/env python3
"""Portable contracts for R127 formal three-request observation, no model imports."""
from pathlib import Path
import ast
import json
import unittest

ROOT=Path(__file__).resolve().parent


class LifecycleContracts(unittest.TestCase):
    def test_all_sources_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))

    def test_single_omni_and_three_full_requests(self):
        source=(ROOT/'generate_lifecycle127.py').read_text()
        tree=ast.parse(source)
        omni=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Omni']
        self.assertEqual(len(omni),1)
        self.assertIn('num_inference_steps=49',source)
        self.assertIn('steps_per_rank\':49',source)
        self.assertIn('182280',source)
        self.assertIn("else:assert pids==first_pids and ids==pipeline_ids",source)

    def test_tp_and_fp32_defaults_unchanged(self):
        source=(ROOT/'generate_lifecycle127.py').read_text()
        for text in ("'tensor_parallel_size':2","'text_encoder_tp_size':8","'ulysses_degree':4","'ring_degree':1"):
            self.assertIn(text,source)
        source=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn('ieee_nonfinite_masks(tensor)',source)
        self.assertIn('count==155',source)
        self.assertIn('49*155',source)
        self.assertIn('with set_default_torch_dtype(torch.float32)',source)
        self.assertIn('torch.npu.matmul.allow_hf32=False',source)

    def test_only_request_audit_counter_is_reset(self):
        source=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn('self.transformer._request_steps127=0',source)
        self.assertIn('self._total_steps127+=1',source)
        self.assertNotIn('self.transformer._checks127=0',source)
        self.assertIn("self.transformer._checks127-before_checks==49*155",source)

    def test_no_r126_candidates_or_shared_stream_change(self):
        source=(ROOT/'launch_lifecycle127.py').read_text()
        self.assertIn("'H3_R126_VAE_OPERAND_CACHE':'0'",source)
        self.assertNotIn("'VLLM_OMNI_910A_SHARED_STREAM':'0'",source)
        obs=(ROOT/'observe_runtime127.py').read_text()
        self.assertNotIn('torch.npu.Stream()',obs)
        self.assertNotIn('set_affinity',obs)
        self.assertNotIn('get_or_create(block)',obs)

    def test_request_scoped_artifacts(self):
        source=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn("request_directory()/f'live_encoder_pid",source)
        self.assertIn("request_directory()/f'final_latents_pid",source)
        self.assertNotIn('GENERATION_ROOT=',source)
        context=(ROOT/'runtime_context127.py').read_text()
        self.assertIn("LABELS={1:'r01_cold',2:'r02_warm',3:'r03_native_reload'}",context)
        self.assertIn("'request_index':_CURRENT",context)

    def test_no_model_bootstrap_for_cann_helpers(self):
        source=(ROOT/'full_bootstrap127.py').read_text()
        self.assertIn("if self.name.startswith('DiffusionWorker-'):install_alignment()",source)
        self.assertIn('os.environ.pop',source)
        source=(ROOT/'sitecustomize.py').read_text()
        self.assertNotIn('from integration_hooks127',source)
        self.assertIn('len(key)==32',source)

    def test_no_module_synchronize_or_transfer_hooks(self):
        source=(ROOT/'observe_runtime127.py').read_text()
        self.assertNotIn('torch.npu.synchronize()',source)
        self.assertNotIn('torch.load',source)
        self.assertNotIn('.cpu()',source)
        self.assertIn('node_counts_are_whole_VMA_not_exact_tensor_pages',source)

    def test_lifecycle_summary_rank_max_and_gate_counts(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('lifecycle_summary_unit',ROOT/'summarize_lifecycle127.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=[{'step_index':i,'elapsed':float(i+rank)} for i in range(49) for rank in range(8)]
        result=module.median_rank_max(rows,'elapsed')
        self.assertEqual(result['first_step_s'],7)
        self.assertEqual(len(result['every_step_rank_max_s']),49)
        self.assertEqual(result['warm_steps_1_to_48_median_s'],31.5)
        with self.assertRaises(AssertionError):module.median_rank_max(rows[:-1],'elapsed')

    def test_summary_uses_actual_requests_and_no_numeric_acceptance(self):
        source=(ROOT/'summarize_lifecycle127.py').read_text()
        self.assertIn("assert result['completed'] and len(result['requests'])==3",source)
        self.assertIn('model_ids==expected_ids',source)
        self.assertIn('==182280',source)
        self.assertIn("'numeric_candidate_automatically_accepted':False",source)
        self.assertNotIn('torch.load',source)

    def test_audit_completion_markers_only_after_file_close(self):
        import importlib.util
        import tempfile
        spec=importlib.util.spec_from_file_location('audit_completion_unit',ROOT/'audit_completion127.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            self.assertIsNone(module.read_complete(path,1))
            for pid in range(1,9):module.publish_complete(path,1,pid,success=True)
            result=module.wait_complete(path,1)
            self.assertEqual(result['pids'],list(range(1,9)))
            with self.assertRaises(AssertionError):module.publish_complete(path,1,1,success=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);module.publish_complete(path,1,1,success=False)
            with self.assertRaises(RuntimeError):module.read_complete(path,1)
        source=(ROOT/'runtime_context127.py').read_text()
        self.assertIn('flush();_CURRENT=0',source)
        self.assertIn('publish_complete(directory,index,os.getpid(),success=success)',source)
        driver=(ROOT/'generate_lifecycle127.py').read_text()
        self.assertIn('completion=wait_complete(out,index,timeout_s=120)',driver)
        self.assertIn("worker_audit_wait_s_outside_generation",driver)

    def test_launcher_driver_collector_roots_match(self):
        for filename in ('launch_lifecycle127.py','generate_lifecycle127.py','collect_lifecycle127.py','compare_native_repeat127.py'):
            source=(ROOT/filename).read_text()
            self.assertIn('engine_lifecycle127c',source)
            self.assertNotIn('engine_lifecycle127b',source)

    def test_initialization_diagnostic_preserves_native_copy(self):
        source=(ROOT/'init_diagnostics127.py').read_text()
        self.assertIn('result=native(params,bufs,pin_memory)',source)
        self.assertIn('bytes_required>storage_bytes',source)
        self.assertNotIn('.copy_(',source)
        self.assertNotIn('torch.npu.Stream',source)
        self.assertIn('faulthandler.enable',source)

    def test_fresh_launch_and_source_pin_guards(self):
        source=(ROOT/'launch_lifecycle127.py').read_text()
        self.assertIn('not SESSION.exists() and not receipt.exists()',source)
        self.assertIn("os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600",source)
        self.assertIn('hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest',source)


if __name__=='__main__':unittest.main()
