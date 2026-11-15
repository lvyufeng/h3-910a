#!/usr/bin/env python3
"""Dependency-free structural guards for this narrow correctness/timing patch."""
import ast
import json
import sys
import types
from unittest.mock import patch
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent

def function(path,cls,name):
    tree=ast.parse(path.read_text())
    scope=tree.body if cls is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in scope if isinstance(n,ast.FunctionDef) and n.name==name)

class ContractTests(unittest.TestCase):
    def test_embedding_uses_collective_return(self):
        fn=function(ROOT/'candidate_encoder.py','MiniMaxH3Qwen3VLVocabParallelEmbedding','forward')
        last=fn.body[-1]
        self.assertIsInstance(last,ast.Return)
        self.assertEqual(ast.unparse(last.value),'self.group.all_reduce(output)')

    def test_row_linear_uses_collective_return(self):
        fn=function(ROOT/'candidate_encoder.py','MiniMaxH3Qwen3VLRowParallelLinear','forward')
        assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='self.group.all_reduce']
        self.assertEqual(len(assignments),1)
        self.assertEqual(ast.unparse(assignments[0].targets[0]),'output_parallel')

    def test_weight_loaders_unchanged(self):
        for cls in ('MiniMaxH3Qwen3VLVocabParallelEmbedding','MiniMaxH3Qwen3VLRowParallelLinear'):
            a=function(ROOT/'source/encoder.py',cls,'weight_loader')
            b=function(ROOT/'candidate_encoder.py',cls,'weight_loader')
            self.assertEqual(ast.dump(a),ast.dump(b))

    def test_cooperative_cancellation_call_unchanged(self):
        def sync_calls(path):
            fn=function(path,None,'minimax_h3_denoise_loop')
            return [ast.unparse(n) for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='check_request_cancellation' and any(k.arg=='synchronize' for k in n.keywords)]
        self.assertEqual(sync_calls(ROOT/'source/denoise_loop.py'),sync_calls(ROOT/'candidate_denoise_loop.py'))

    def test_complete_timer_follows_callback_and_cancellation(self):
        text=(ROOT/'candidate_denoise_loop.py').read_text()
        callback=text.index('on_step(step, video_rows, audio_rows)')
        sync=text.index('check_request_cancellation(synchronize=True)',callback)
        timer=text.index('_complete_end = _time_d.perf_counter()',sync)
        self.assertLess(callback,sync); self.assertLess(sync,timer)
        self.assertNotIn('projected_49steps',text)

    def test_measurement_mode_explicitly_drains_before_timer_end(self):
        fn=function(ROOT/'candidate_denoise_loop.py',None,'minimax_h3_denoise_loop')
        guarded=[]
        for node in ast.walk(fn):
            if isinstance(node,ast.If) and ast.unparse(node.test)=='_perf_d':
                calls=[n for child in node.body for n in ast.walk(child) if isinstance(n,ast.Call)
                       and ast.unparse(n.func)=='current_omni_platform.synchronize']
                if calls: guarded.append(node)
        self.assertEqual(len(guarded),1)
        text=(ROOT/'candidate_denoise_loop.py').read_text()
        self.assertLess(text.index('current_omni_platform.synchronize()'),text.index('_complete_end = _time_d.perf_counter()'))

    def test_actual_cancellation_helper_does_not_sync_successful_steps(self):
        fn=function(ROOT/'source/cancellation.py',None,'check_request_cancellation')
        for signals in (None,(types.SimpleNamespace(buf=bytearray([0])),)):
            ns={'_current_signals':types.SimpleNamespace(get=lambda:signals)}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual_cancel_helper','exec'),ns)
            stub=types.SimpleNamespace(distributed=types.SimpleNamespace(is_initialized=lambda:False))
            # No platform module is provided: an unconditional platform sync here
            # would fail instead of silently passing a fake five-second drain.
            with patch.dict(sys.modules,{'torch':stub}):
                self.assertIsNone(ns['check_request_cancellation'](synchronize=True))

    def test_four_step_artifact_does_not_claim_fl2va_support(self):
        meta=json.loads((ROOT/'source/fused_lora.json').read_text())
        self.assertEqual(meta['lora_metadata']['tasks'],'t2va')
        self.assertEqual(meta['steps'],4)
        release=json.loads((ROOT/'source/original_fl2va_model_index.json').read_text())
        self.assertIn('fl2va',release['_minimax_h3']['tasks'])

    def test_native_request_uses_real_sampling_field_not_extra_args(self):
        text=(ROOT/'e2e_native49_probe.py').read_text()
        self.assertIn('"num_inference_steps": "49"',text)
        self.assertNotIn('json.dumps({"task": "fl2va", "num_inference_steps"',text)
        self.assertIn('h3_round53_native49_fp16_model',text)

if __name__=='__main__': unittest.main(verbosity=2)
