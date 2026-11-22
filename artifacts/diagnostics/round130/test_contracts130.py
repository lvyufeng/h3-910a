#!/usr/bin/env python3
"""Portable source contracts; actual mathematical diagnostics run on owned host."""
import ast
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def test_sources_parse(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),filename=str(p))

    def test_observer_calls_original_returns_only(self):
        source=(ROOT/'projection_observer130.py').read_text()
        self.assertIn('result=mm(a,b,*args,**kwargs)',source)
        self.assertIn('result=add(destination,source,*args,**kwargs)',source)
        self.assertIn('return result',source)
        self.assertNotIn('result.copy_',source)

    def test_snapshots_before_mutation_and_immediate_output(self):
        source=(ROOT/'projection_observer130.py').read_text()
        self.assertLess(source.index('before=self.cpu(destination)'),source.index('result=add(destination'))
        self.assertIn('self.cpu(result)  # Snapshot before caller can mutate',source)
        self.assertIn('contiguous().clone()',source)

    def test_scoped_functions_restore_and_check_ownership(self):
        source=(ROOT/'projection_observer130.py').read_text()
        self.assertIn('torch.mm is matrix and torch.Tensor.add_ is addition',source)
        self.assertIn('torch.mm=mm;torch.Tensor.add_=add',source)
        self.assertIn('threading.get_native_id()==self.thread',source)

    def test_real_TP_and_original_guard(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertIn('tensor_parallel_size=2,sequence_parallel_size=4',source)
        self.assertIn('install_time_projection_guard123(holder)',source)
        self.assertIn('parameters_before',source);self.assertIn('parameters_after',source)
        self.assertNotIn('allow_hf32=True',source)

    def test_no_model_sampling_or_candidate(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertNotIn('.generate(',source);self.assertNotIn('NativeCopyAdapter',source)
        self.assertNotIn('GPU_corrected',source)
        self.assertIn('native_worker_input_step25_rank',source)

    def test_four_GEMMs_and_scaled_adds_counted(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertIn("('mm',0),('mm',1),('add_',0),('mm',2),('add_',1),('mm',3),('add_',2)",source)
        self.assertIn('[2.0**-12,2.0**-12,2.0**-24]',source)
        self.assertIn("if fixed_bias is not None:expected_ops.append(('add_',3))",source)
        self.assertIn('actual_local_bias_present',source)
        self.assertIn('fixed_operand_repeats',source)

    def test_original_instrumented_restored(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertIn("report['native'].append",source)
        self.assertIn("report['observed'].append",source)
        self.assertIn("report['restored'].append",source)
        self.assertIn('global_functions_restored',source)

    def test_fixed_replays_use_original_operations(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertIn('output=original_mm(a,b,**kwargs)',source)
        self.assertIn('output=original_add(a.clone(),b,**kwargs)',source)
        self.assertIn('same_inputs_before_after',source)

    def test_determinism_probe_restores_private_flags(self):
        source=(ROOT/'deterministic_probe130.py').read_text()
        self.assertIn('torch.use_deterministic_algorithms(initial_flag,warn_only=initial_warn)',source)
        self.assertIn("'candidate_policy_adopted':False",source)
        self.assertIn("('deterministic',True)",source)
        self.assertIn('torch.mm(a,b,out_dtype=torch.float32)',source)
        self.assertIn("'versus_native_first'",source)
        self.assertNotIn('set_option(',source)

    def test_snapshot_sync_not_claimed_fix(self):
        source=(ROOT/'time_stages130.py').read_text()
        self.assertIn('capture_sync_is_diagnostic_not_lossless_fix',source)
        observer=(ROOT/'projection_observer130.py').read_text()
        self.assertIn('may mask a scheduling hazard',observer)


if __name__=='__main__':unittest.main()
