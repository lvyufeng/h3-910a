#!/usr/bin/env python3
"""Portable R126 execution-source contracts; no Torch, device or model imports."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import struct
import unittest

ROOT=Path(__file__).resolve().parent


def parsed(path):
    return ast.parse((ROOT/path).read_text())


def calls(tree,name):
    result=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            f=n.func
            if (isinstance(f,ast.Attribute) and f.attr==name) or (isinstance(f,ast.Name) and f.id==name):result.append(n)
    return result


class ExecutionContracts(unittest.TestCase):
    def test_all_half_words_reference_spec(self):
        bad=boundary=0
        for bits in range(65536):
            magnitude=bits&0x7fff
            exponent=magnitude&0x7c00;mantissa=magnitude&0x03ff
            old_bad=exponent==0x7c00
            new_bad=magnitude>=0x7c00
            self.assertEqual(old_bad,new_bad)
            finite_value=struct.unpack('<e',struct.pack('<H',bits))[0]
            old_boundary=abs(finite_value)==65504.0
            new_boundary=magnitude==0x7bff
            self.assertEqual(old_boundary,new_boundary)
            bad+=new_bad;boundary+=new_boundary
        self.assertEqual((bad,boundary),(2048,2))
    def test_pool_profile_is_not_dead_template_code(self):
        t=parsed('benchmark_weight_pool_profile126.py')
        self.assertEqual(len(calls(t,'begin_profile')),1)
        self.assertEqual(len(calls(t,'end_profile')),1)
        self.assertTrue(calls(t,'uninstall'))
    def test_native_pool_uses_explicit_independent_stream(self):
        source=(ROOT/'immutable_weight_pool126.py').read_text()
        self.assertIn('torch.npu.Stream() if asynchronous else self.compute_stream',source)
        self.assertIn('slot_consumed',source)
        self.assertIn('wait_event(self.ready[owner])',source)
        self.assertIn('self.prepared.add(index)',source)
        self.assertIn('self.prepared.discard(index)',source)
        self.assertIn('self.compute_rows==list(range(50))',source)
        self.assertIn('result=original(module,*args,**kwargs)',source)
    def test_VAE_cache_keeps_original_parameters(self):
        t=parsed('vae/autocast_operand_cache126.py')
        assignments=[target for n in ast.walk(t) if isinstance(n,ast.Assign) for target in n.targets]
        self.assertFalse(any(isinstance(a,ast.Attribute) and a.attr in ('weight','bias','data') for a in assignments))
        self.assertTrue(calls(t,'current_stream'))
        source=(ROOT/'vae/autocast_operand_cache126.py').read_text()
        self.assertIn('native_self_repeat_words_equal',source)
        self.assertIn('return reference',source)
    def test_actual_FA_profile_counts_launcher(self):
        source=(ROOT/'actual_attention_profile126.py').read_text()
        self.assertIn("'fa910a_launch','FA.ACL_launch'",source)
        self.assertIn("self.counts['FA.ACL_launch']==50",source)
        self.assertIn("[1,7,41792,128]",source)
    def test_proof_hooks_are_not_clean_speed_hooks(self):
        source=(ROOT/'vae/benchmark_operand_cache126.py').read_text()
        self.assertIn('handle.remove()',source)
        self.assertIn("clean_case('native_before',False)",source)
        self.assertIn('cache.begin_proof()',source)
        self.assertIn('aligned_local_tile_rounds',source)
        self.assertIn('global_count=int(max_count.cpu())',source)
    def test_generation_selective_bootstrap_is_worker_only(self):
        source=(ROOT/'full_bootstrap126.py').read_text()
        self.assertIn("self.name.startswith('DiffusionWorker-')",source)
        self.assertNotIn("authkey=b''",source)
        self.assertIn('len(key)==32',source)
    def test_full_regression_requires_all_worker_checks_and_cache_hits(self):
        source=(ROOT/'generate_full_NPU126.py').read_text()
        self.assertIn('set(counts.values())=={49}',source)
        self.assertIn("row['actual_Half_matrix_modules_this_step']==155",source)
        self.assertIn("len(cache_events)==8",source)
        self.assertIn("'new_combined_configuration_quality_accepted']=False",source)
    def test_equivalent_safety_has_no_value_rewrite(self):
        t=parsed('equivalent_safety126.py')
        self.assertFalse(calls(t,'nan_to_num') or calls(t,'clamp') or calls(t,'copy_'))
    def test_versioned_receipts_do_not_include_model_weights(self):
        for pattern in ('*.pt','*.npz','*.bin','*.safetensors'):
            self.assertEqual(list(ROOT.rglob(pattern)),[],pattern)

if __name__=='__main__':unittest.main()
