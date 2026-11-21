#!/usr/bin/env python3
"""Portable R128 source/ownership tests; hardware evidence is separate."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import sys
import threading
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from owner_stream128 import OwnedCopyStream128,StreamOwnershipError
from native_copy_adapter128 import NativeCopyAdapter128


class Tensor:
    def __init__(self,data):self.data=data
    def pin_memory(self):return self
    def copy_(self,other,non_blocking=True):self.data=other.data;return self
    def cpu(self):return self
    def contiguous(self):return self
    def view(self,dtype):return self


class Stream:
    def __init__(self,number):self.npu_stream=number;self.waits=[];self.syncs=0
    def wait_stream(self,stream):self.waits.append(('stream',stream.npu_stream))
    def wait_event(self,event):self.waits.append(('event',event.recorded))
    def synchronize(self):self.syncs+=1


class Event:
    def __init__(self):self.recorded=False;self.done=False
    def record(self,stream):self.recorded=True
    def query(self):assert self.recorded;return self.done
    def synchronize(self):assert self.recorded;self.done=True


class NPU:
    def __init__(self):self.current=Stream(0);self.device=0;self.accesses=0
    def current_device(self):self.accesses+=1;return self.device
    def current_stream(self):self.accesses+=1;return self.current
    def Stream(self,device):return Stream(1)
    def Event(self):return Event()
    def stream(self,stream):
        npu=self
        class Context:
            def __enter__(self):self.old=npu.current;npu.current=stream
            def __exit__(self,*args):npu.current=self.old
        return Context()


class Torch:
    float16='half';uint8='byte'
    def __init__(self):self.npu=NPU()
    def arange(self,n,dtype):return Tensor(list(range(n)))
    def empty(self,n,dtype,device):return Tensor([])
    def equal(self,a,b):return a.data==b.data


class Tests(unittest.TestCase):
    def test_all_sources_parse(self):
        for path in ROOT.glob('*.py'):ast.parse(path.read_text(),filename=str(path))

    def test_owner_capability_retains_warmup_probe(self):
        backend=Torch();owner=OwnedCopyStream128(backend,device=0)
        self.assertTrue(owner.facts()['strict_allocator_warmup_completed'])
        self.assertIsNotNone(owner._probe);self.assertNotEqual(owner.compute,owner.copy)
        self.assertEqual(owner.compute.syncs,1);self.assertEqual(owner.copy.syncs,1)
        owner.close();self.assertIsNone(owner._probe)

    def test_wrong_thread_fails_before_runtime_touch(self):
        backend=Torch();owner=OwnedCopyStream128(backend,device=0);start=backend.npu.accesses;errors=[]
        def alien():
            try:owner.require_owner()
            except StreamOwnershipError:errors.append(True)
        thread=threading.Thread(target=alien);thread.start();thread.join()
        self.assertEqual(errors,[True]);self.assertEqual(backend.npu.accesses,start);owner.close()

    def test_device_change_refused(self):
        backend=Torch();owner=OwnedCopyStream128(backend,device=0);backend.npu.device=1
        with self.assertRaises(StreamOwnershipError):owner.require_owner()
        backend.npu.device=0;owner.close()

    def test_compute_stream_change_refused(self):
        backend=Torch();owner=OwnedCopyStream128(backend,device=0);backend.npu.current=Stream(3)
        with self.assertRaises(StreamOwnershipError):owner.require_owner()
        backend.npu.current=owner.compute;owner.close()

    def test_warmup_error_not_suppressed(self):
        backend=Torch();backend.empty=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('allocator unavailable'))
        with self.assertRaisesRegex(RuntimeError,'allocator unavailable'):OwnedCopyStream128(backend,device=0)

    def test_closed_capability_refused(self):
        owner=OwnedCopyStream128(Torch(),device=0);owner.close();owner.close()
        with self.assertRaises(StreamOwnershipError):owner.require_owner()

    def test_no_global_factory_or_sdk_mutation(self):
        source=(ROOT/'owner_stream128.py').read_text()+(ROOT/'native_copy_adapter128.py').read_text()
        self.assertNotIn('platform.Stream =',source);self.assertNotIn('VLLM_OMNI_910A_SHARED_STREAM',source)
        self.assertNotIn('except Exception:',source)

    def test_adapter_only_changes_existing_instances(self):
        source=(ROOT/'native_copy_adapter128.py').read_text()
        self.assertIn('setattr(h,name,method)',source)
        self.assertIn('original(non_blocking=non_blocking)',source)
        self.assertNotIn('ImmutableWeightPool126',source)
        self.assertNotIn('.copy_(',source)
        self.assertIn('len(self.blocks)==len(self.hooks)==50',source)

    def test_storage_not_parameter_retention(self):
        source=(ROOT/'native_copy_adapter128.py').read_text()
        self.assertIn('else t.data).detach()',source)
        self.assertIn('done.record(self.capability.compute)',source)
        self.assertIn('self.retired.append((done,held,index))',source)

    def test_unrecorded_event_is_not_valid_fence(self):
        event=Event()
        with self.assertRaises(AssertionError):event.query()
        event.record(Stream(0));self.assertFalse(event.query())
        event.synchronize();self.assertTrue(event.query())

    def test_explicit_retirement_backpressure_is_bounded(self):
        adapter=NativeCopyAdapter128.__new__(NativeCopyAdapter128)
        class Storage:
            _cdata=1
            def nbytes(self):return 64
        class Held:
            def untyped_storage(self):return Storage()
        events=[Event() for _ in range(3)]
        for event in events:event.record(Stream(0))
        adapter.retired=[(event,[Held()],index) for index,event in enumerate(events)]
        adapter.max_retired=2;adapter.retirement_backpressure_count=0;adapter.retirement_backpressure_host_s=0
        adapter.peak_retired_blocks=0;adapter.peak_retired_bytes=0
        adapter._reap()
        self.assertEqual(len(adapter.retired),2);self.assertEqual(adapter.retirement_backpressure_count,1)
        self.assertTrue(events[0].done);self.assertEqual(adapter.peak_retired_bytes,128)

    def test_rollback_drains_before_storage_release(self):
        source=(ROOT/'native_copy_adapter128.py').read_text().split('    def rollback(self):')[1]
        self.assertLess(source.index('self.capability.drain()'),source.index('clear_block_storage('))
        self.assertIn("self.state='rollback_failed';raise",source)
        self.assertIn("Refuse third-party instance-hook overwrite",source)
        self.assertIn('self.hooks[0].prefetch_layer(non_blocking=False)',source)

    def test_formal_gate_uses_own_live_inputs_not_teacher_features(self):
        source=(ROOT/'formal_replay128.py').read_text()
        self.assertIn("clone_tree(kwargs,torch,'cpu')",source)
        self.assertNotIn('GPU_corrected_pipeline_context',source)
        self.assertIn("'teacher_hidden_or_replay_outputs_used_by_sampler':False",source)
        self.assertIn("len(prepared)==104",source)
        self.assertIn('self.old.apply(ReadOnlyLayer(),x,bias)',source)

    def test_candidate49_waits_for_actual_bit_and_speed_gates(self):
        source=(ROOT/'formal_replay128.py').read_text()
        self.assertIn('all_equal and gain is not None and gain>=0.01',source)
        driver=(ROOT/'generate_lifecycle127.py').read_text()
        self.assertIn("candidate_formal_generation_allowed",driver)
        self.assertIn("'candidate_49_generation_not_launched'",driver)

    def test_nontrajectory_checks_not_counted_as_sampler_steps(self):
        source=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn('if self._benchmark_active128:self._benchmark_checks128+=1',source)
        self.assertIn('else:self._checks127+=1',source)
        self.assertIn('self.transformer._checks127-before_checks==49*155',source)
        self.assertIn('Replay overwrote the original prediction',source)

    def test_runtime_context_import_matches_exact_new_session(self):
        import os
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();session=root/'engine_overlap128a';session.mkdir()
            source=(ROOT/'runtime_context127.py').read_text().replace("ROOT=Path('/tmp/h3_overlap12801')",f'ROOT=Path({str(root)!r})')
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(session)}):
                namespace={'__name__':'runtime_context_unit'};exec(compile(source,'runtime_context127.py','exec'),namespace)
                self.assertEqual(namespace['SESSION'],session)
            invalid=root/'engine_lifecycle127c';invalid.mkdir()
            with patch.dict(os.environ,{'H3_R127_SESSION_OUT':str(invalid)}):
                with self.assertRaises(AssertionError):exec(compile(source,'runtime_context127.py','exec'),{'__name__':'invalid_runtime_context_unit'})

    def test_formal_ipc_and_driver_roots_match(self):
        for name in ('full_bootstrap127.py','generate_lifecycle127.py','launch_lifecycle127.py','runtime_context127.py'):
            source=(ROOT/name).read_text()
            self.assertIn('/tmp/h3_overlap12801',source)
            self.assertNotIn('/tmp/h3_runtime12701',source)
        context=(ROOT/'runtime_context127.py').read_text()
        self.assertIn("SESSION.name=='engine_overlap128a'",context)
        self.assertNotIn("engine_lifecycle127",context)
        driver=(ROOT/'generate_lifecycle127.py').read_text()
        self.assertEqual(driver.count('omni=Omni('),1)
        self.assertIn("'tensor_parallel_size':2,'text_encoder_tp_size':8",driver)
        self.assertIn('num_inference_steps=49',driver)

    def test_cohort_phase_wraps_setup_and_return_to_sampler(self):
        source=(ROOT/'formal_replay128.py').read_text()
        for phase in ('collect_native_ring','master_before_SHA','clone_live_input_CPU','save_native_input_fixture',
                      'all_master_dtype_and_native_streams_restored','persist_completed_gate_before_return_to_sampler'):
            self.assertIn(phase,source)
        integration=(ROOT/'integration_hooks127.py').read_text()
        self.assertIn('original_prediction_preserved_before_sampler26',integration)
        guard=(ROOT/'cohort128.py').read_text()
        self.assertIn('_get_default_store()',guard)
        self.assertIn('self.store.wait(keys,self.timeout)',guard)

    def test_no_rank_local_numeric_raise_mid_byteproof_forward(self):
        source=(ROOT/'native_copy_adapter128.py').read_text()
        self.assertNotIn("assert equal,'Native independent-stream",source)
        self.assertIn('entire_native_flat_master_raw_words_equal',source)
        gate=(ROOT/'formal_replay128.py').read_text()
        self.assertIn("all(r['entire_native_flat_master_raw_words_equal'] for r in active.byte_proofs)",gate)

    def test_mirror_validates_all104_before_commit(self):
        source=(ROOT/'formal_replay128.py').read_text().split('class MirrorLeafProof128:')[1].split('def benchmark_in_worker')[0]
        self.assertLess(source.index('assert len(prepared)==104'),source.index('layer.quant_method=wrapper'))
        self.assertIn('except BaseException:self.close();raise',source)

    def test_fatal_monitor_is_private_and_created_before_model(self):
        launcher=(ROOT/'launch_lifecycle127.py').read_text()
        self.assertLess(launcher.index('failure_watch=prepare_failure_watch128()'),launcher.index('process=subprocess.Popen'))
        monitor=(ROOT/'cohort128.py').read_text()
        self.assertIn('os.getpgid(process.pid)==process.pid',monitor)
        self.assertIn('H3_R127_SESSION_OUT=/tmp/h3_overlap12801/engine_overlap128a',monitor)
        self.assertIn('start_new_session=True',launcher)
        self.assertNotIn('sleep(',monitor)

    def test_native_state_restoration_is_not_output_repeatability(self):
        helper=(ROOT/'formal_replay128.py').read_text();driver=(ROOT/'generate_lifecycle127.py').read_text()
        self.assertIn("report['native_hook_state_verified']=True",helper)
        self.assertIn("report['restored_output_bits_equal']=restored",helper)
        self.assertIn("r['completed'] and r['native_hook_state_verified']",driver)
        self.assertNotIn("r['completed'] and r['restored']",driver)
        self.assertIn('Blocked: unchanged native baseline is not bit-repeatable',helper)

    def test_synthetic_gates_are_not_real_model_claims(self):
        source=(ROOT/'test_worker_stream128.py').read_text()+(ROOT/'test_native_hooks128.py').read_text()
        self.assertIn('not_model_performance_evidence',source)
        self.assertIn('not_original_155check_or_real_H3_generation_gate',source)


if __name__=='__main__':unittest.main()
