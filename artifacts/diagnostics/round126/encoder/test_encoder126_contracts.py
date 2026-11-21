"""Stdlib-only mocks/AST contracts. These are NOT NPU numerical/performance gates."""
import ast
from itertools import count
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import native_geometry_memo126 as M
from stage_profile126 import EncoderStageProbe126, measure_complete

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent.parent / 'round124' / 'NPU_encoder_FP16_executed124.py'
POINTERS = count(100, 100)


class Device:
    def __init__(self, value):
        self.value = value
        self.type = value.split(':')[0]
    def __str__(self): return self.value
    def __eq__(self, other): return isinstance(other, Device) and self.value == other.value


class Tensor:
    def __init__(self, shape, dtype, *, device='npu:0', payload=b'original', rows=None, version=0):
        self.shape, self.dtype, self.device = tuple(shape), dtype, Device(device)
        self.payload, self.rows, self.version = payload, rows, version
        self.pointer, self.value_reads = next(POINTERS), 0
    @property
    def _version(self):
        if self.version is None:
            raise RuntimeError('Inference tensors do not track version counter')
        return self.version
    def untyped_storage(self): return self
    def data_ptr(self): return self.pointer
    def storage_offset(self): return 0
    def stride(self):
        result, size = [], 1
        for value in reversed(self.shape): result.append(size); size *= value
        return tuple(reversed(result))
    def detach(self): return self
    def contiguous(self): return self
    def clone(self): return Tensor(self.shape, self.dtype, device=str(self.device), payload=self.payload, rows=self.rows)
    def numel(self):
        count_ = 1
        for value in self.shape: count_ *= value
        return count_
    def element_size(self): return 2 if self.dtype == 'torch.float16' else 4
    def numpy(self): return SimpleNamespace(tobytes=lambda: self.payload)
    def tolist(self):
        self.value_reads += 1
        return [list(row) for row in self.rows]


class Vision:
    def __init__(self):
        self.pos_embed = SimpleNamespace(weight=Tensor((2304, 1152), 'torch.float16'))
        self.rotary_pos_emb = SimpleNamespace(_inv_freq=Tensor((18,), 'torch.float32', device='cpu'), dim=36, theta=10000.)
        self.num_grid_per_side, self.spatial_merge_size, self.patch_size = 48, 2, 16
        self.config = SimpleNamespace(hidden_size=1152, num_heads=16, intermediate_size=4304,
                                      depth=27, num_position_embeddings=2304, spatial_merge_size=2)
        self.calls = []
        self.live_calls = 0
    def fast_pos_embed_interpolate(self, grid_thw):
        self.calls.append('position')
        rows = grid_thw.tolist()
        tokens = sum(t*h*w for t,h,w in rows)
        return Tensor((tokens,1152), 'torch.float16', device=str(grid_thw.device),
                      payload=b'position:' + self.pos_embed.weight.payload + repr(rows).encode())
    def rot_pos_emb(self, grid_thw):
        self.calls.append('rotary')
        rows = grid_thw.tolist()
        tokens = sum(t*h*w for t,h,w in rows)
        return Tensor((tokens,36), 'torch.float32', device=str(grid_thw.device),
                      payload=b'rotary:' + self.rotary_pos_emb._inv_freq.payload + repr(rows).encode())
    def forward(self, pixels, grid_thw):
        self.live_calls += 1
        return {'fresh_pixel_marker': pixels, 'position': self.fast_pos_embed_interpolate(grid_thw),
                'rotary': self.rot_pos_emb(grid_thw)}
    def _apply(self, function):
        function(self.pos_embed.weight)
        return self


class Encoder:
    def __init__(self):
        self.vision = Vision()
    def load_weights(self, payload):
        self.vision.pos_embed.weight.payload = payload  # Intentional unversioned .data-style mutation.
        return self
    def load_to_device(self): return self
    def offload_to_cpu(self): return self
    def _apply(self, function):
        return self.vision._apply(function)


def runtime():
    return {'default_dtype': 'torch.float32', 'default_device': 'cpu', 'autocast_enabled': False,
            'autocast_dtype': 'torch.float16', 'matmul_allow_hf32': False, 'conv_allow_hf32': False,
            'stream_id': 1}


def grid(rows=((1,48,84),(1,48,84)), version=None):
    return Tensor((len(rows),3), 'torch.int64', rows=rows, version=version)


class CacheContracts(unittest.TestCase):
    def setup_cache(self, **kwargs):
        encoder = Encoder()
        state = runtime()
        # Public source pin remains tested below. Only the fake methods' origin
        # check is mocked; no Torch/native baseline function is executed here.
        with patch.object(M, '_verify_bound_source', return_value=None):
            cache = M.install(encoder, BASELINE, lambda: dict(state), **kwargs)
        cache.set_enabled(True)
        self.addCleanup(self.safe_uninstall, cache)
        return encoder, cache, state
    def safe_uninstall(self, cache):
        for owner,name,original,wrapper in cache._bindings:
            setattr(owner,name,wrapper)
        cache.uninstall()
    def test_source_file_is_exact_R124(self):
        proof = M.verify_source_file(BASELINE)
        self.assertEqual(proof['sha256'], M.BASELINE_SHA256)
        self.assertEqual(set(proof['method_sha256']), {'forward',*M.GEOMETRY_METHODS})
    def test_unverified_fake_methods_rejected(self):
        with self.assertRaises(M.GeometryMemoError):
            M.install(Encoder(), BASELINE, runtime)
    def test_no_Torch_import_or_trig_reimplementation(self):
        tree = ast.parse((HERE/'native_geometry_memo126.py').read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        imports += [name.name for node in ast.walk(tree) if isinstance(node,ast.Import) for name in node.names]
        self.assertFalse(any(name and ('torch' in name or 'numpy' in name) for name in imports))
        self.assertFalse(any(isinstance(node,ast.Attribute) and node.attr in ('sin','cos','matmul','linear') for node in ast.walk(tree)))
    def test_first_calls_original_and_hits_still_run_live_forward(self):
        encoder,cache,state = self.setup_cache()
        g=grid(); first=encoder.vision.forward('pixelsA',g); second=encoder.vision.forward('pixelsB',g)
        self.assertEqual(encoder.vision.live_calls,2)
        self.assertEqual(encoder.vision.calls,['position','rotary'])
        self.assertEqual(second['fresh_pixel_marker'],'pixelsB')
        self.assertEqual(first['position'].payload,second['position'].payload)
        self.assertEqual(first['rotary'].payload,second['rotary'].payload)
        self.assertEqual(g.value_reads,4)  # first key + two original methods; hit key once.
        self.assertEqual(cache.stats['grid_value_reads'],2)
        self.assertEqual(cache.snapshot()['cached_geometry_bytes'],19740672)
    def test_hit_returns_owned_fresh_clones_not_shared_scratch(self):
        encoder,cache,state=self.setup_cache(); g=grid()
        a=encoder.vision.forward('a',g); b=encoder.vision.forward('b',g); c=encoder.vision.forward('c',g)
        masters=[e['master'] for e in cache._entries.values()]
        pointers=[a['position'].pointer,b['position'].pointer,c['position'].pointer,*[m.pointer for m in masters]]
        self.assertEqual(len(pointers),len(set(pointers)))
        b['position'].payload=b'poison'; b['position'].version+=1
        self.assertEqual(encoder.vision.forward('d',g)['position'].payload,a['position'].payload)
    def test_private_version_mutation_is_rejected(self):
        encoder,cache,state=self.setup_cache(); g=grid(); encoder.vision.forward('a',g)
        next(iter(cache._entries.values()))['master'].version+=1
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('b',g)
        self.assertEqual(cache.snapshot()['entries'],0)
    def test_new_grid_object_same_values_can_hit(self):
        encoder,cache,state=self.setup_cache(); encoder.vision.forward('a',grid())
        encoder.vision.forward('b',grid())
        self.assertEqual(cache.stats['hits'],2)
    def test_unversioned_grid_value_change_recomputes(self):
        encoder,cache,state=self.setup_cache(); g=grid(); encoder.vision.forward('a',g)
        g.rows=((1,24,84),(1,24,84)); result=encoder.vision.forward('b',g)
        self.assertEqual(result['position'].shape,(4032,1152))
        self.assertEqual(len(encoder.vision.calls),4)
    def test_weight_version_change_recomputes_after_epoch_change(self):
        encoder,cache,state=self.setup_cache(); g=grid(); encoder.vision.forward('a',g)
        encoder.vision.pos_embed.weight.version+=1; encoder.vision.pos_embed.weight.payload=b'newweight'
        out=encoder.vision.forward('b',g)
        self.assertIn(b'newweight',out['position'].payload)
        self.assertEqual(cache.stats['invalidations'],1)
        self.assertEqual(cache.stats['misses'],4)
    def test_data_loader_is_explicitly_invalidated(self):
        encoder,cache,state=self.setup_cache(); g=grid(); encoder.vision.forward('a',g)
        version=encoder.vision.pos_embed.weight.version
        encoder.load_weights(b'new .data')
        self.assertEqual(encoder.vision.pos_embed.weight.version,version)
        self.assertEqual(cache.snapshot()['entries'],0)
        self.assertIn(b'new .data',encoder.vision.forward('b',g)['position'].payload)
    def test_external_stager_notify_drops_masters(self):
        encoder,cache,state=self.setup_cache(); encoder.vision.forward('a',grid())
        cache.invalidate('lead explicit direct stager rebind')
        self.assertEqual(cache.snapshot()['entries'],0)
    def test_noop_load_and_offload_conservatively_clear(self):
        encoder,cache,state=self.setup_cache(); g=grid(); encoder.vision.forward('a',g)
        encoder.load_to_device();self.assertEqual(cache.snapshot()['entries'],0)
        encoder.vision.forward('b',g);encoder.offload_to_cpu();self.assertEqual(cache.snapshot()['entries'],0)
    def test_inv_freq_unversioned_value_change_recomputes(self):
        encoder,cache,state=self.setup_cache();g=grid();encoder.vision.forward('a',g)
        encoder.vision.rotary_pos_emb._inv_freq.payload=b'newfloatbits'
        self.assertIn(b'newfloatbits',encoder.vision.forward('b',g)['rotary'].payload)
        self.assertEqual(cache.stats['invalidations'],1)
    def test_runtime_default_dtype_and_stream_changes_invalidate(self):
        encoder,cache,state=self.setup_cache();g=grid();encoder.vision.forward('a',g)
        state['default_dtype']='torch.float16';encoder.vision.forward('b',g)
        state['stream_id']=2;encoder.vision.forward('c',g)
        self.assertEqual(cache.stats['invalidations'],2)
        self.assertEqual(cache.stats['misses'],6)
    def test_incomplete_runtime_or_wrong_device_refused(self):
        encoder,cache,state=self.setup_cache();state.pop('stream_id')
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('a',grid())
        encoder,cache,state=self.setup_cache();g=grid();g.device=Device('npu:1')
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('a',g)
    def test_apply_dtype_change_clears_then_rejects_BF16(self):
        encoder,cache,state=self.setup_cache();g=grid();encoder.vision.forward('a',g)
        encoder._apply(lambda tensor:setattr(tensor,'dtype','torch.bfloat16'))
        self.assertEqual(cache.snapshot()['entries'],0)
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('b',g)
    def test_config_change_refused(self):
        encoder,cache,state=self.setup_cache();g=grid();encoder.vision.forward('a',g)
        encoder.vision.config.hidden_size=1153
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('b',g)
    def test_budget_bypass_keeps_genuine_original_output(self):
        encoder,cache,state=self.setup_cache(max_bytes=1);g=grid()
        encoder.vision.forward('a',g);encoder.vision.forward('b',g)
        self.assertEqual(cache.snapshot()['entries'],0)
        self.assertEqual(cache.stats['budget_bypasses'],4)
        self.assertEqual(cache.stats['private_clones'],0)
    def test_LRU_does_not_reuse_other_grid(self):
        encoder,cache,state=self.setup_cache(max_entries=2);g=grid()
        first=encoder.vision.forward('a',g)
        encoder.vision.forward('b',grid(((1,24,84),(1,24,84))))
        again=encoder.vision.forward('c',g)
        self.assertEqual(again['position'].payload,first['position'].payload)
        self.assertEqual(cache.stats['misses'],6)
        self.assertLessEqual(cache.snapshot()['entries'],2)
    def test_disabled_direct_geometry_calls_delegate(self):
        encoder,cache,state=self.setup_cache();cache.set_enabled(False)
        encoder.vision.forward('a',grid())
        encoder.vision.rot_pos_emb(grid())
        self.assertEqual(cache.snapshot()['entries'],0)
        self.assertEqual(cache.stats['grid_value_reads'],0)
        self.assertEqual(cache.stats['original_geometry_calls'],3)
    def test_ownership_replacement_refuses_atomic_uninstall(self):
        encoder,cache,state=self.setup_cache();g=grid();encoder.vision.forward('a',g)
        encoder.vision.rot_pos_emb=lambda grid:None
        with self.assertRaises(M.GeometryMemoError):encoder.vision.forward('b',g)
        with self.assertRaises(M.GeometryMemoError):cache.uninstall()
        self.assertIn('fast_pos_embed_interpolate',vars(encoder.vision))
    def test_uninstall_restores_class_methods_and_drops_all_masters(self):
        encoder,cache,state=self.setup_cache();encoder.vision.forward('a',grid());cache.uninstall()
        self.assertEqual(cache.snapshot()['entries'],0)
        self.assertNotIn('forward',vars(encoder.vision))
        self.assertNotIn('load_weights',vars(encoder))
        with self.assertRaises(M.GeometryMemoError):cache.set_enabled(True)


class StageContracts(unittest.TestCase):
    def test_complete_timer_includes_native_D2H_excludes_start_and_proof(self):
        clock=[0.]; pending=[0.]
        def barrier():clock[0]+=100.
        def drain():clock[0]+=pending[0];pending[0]=0.
        def call():clock[0]+=7.;pending[0]+=5.;return 'native output includesD2H'
        def proof(output):clock[0]+=1000.;return output
        out,seconds,evidence=measure_complete(call,start_barrier=barrier,drain=drain,clock=lambda:clock[0],proof=proof)
        self.assertEqual(seconds,12.)
        self.assertEqual(clock[0],1112.)
        self.assertEqual(out,evidence)
    def test_nested_intervals_label_parent_and_do_not_claim_clean(self):
        clock=[0.];p=EncoderStageProbe126(clock=lambda:clock[0]);p.enabled=True
        with p.stage('encode_prompt'):
            clock[0]+=1.
            with p.stage('vision'):clock[0]+=2.
            clock[0]+=1.
        parent=next(r for r in p.rows if r['stage']=='encode_prompt')
        child=next(r for r in p.rows if r['stage']=='vision')
        self.assertEqual(parent['inclusive_host_wall_s'],4.)
        self.assertEqual(child['parent_id'],parent['id'])
        self.assertTrue(parent['contains_child_intervals'])
        self.assertFalse(p.report()['clean_performance'])
    def test_probe_disabled_and_exception_cleanup(self):
        p=EncoderStageProbe126()
        self.assertEqual(p.call('native',lambda:5),5);self.assertEqual(p.rows,[])
        p.enabled=True
        with self.assertRaises(ValueError):
            with p.stage('failing'):raise ValueError('native failure')
        self.assertTrue(p.rows[0]['failure']);self.assertEqual(p._stack,[])
    def test_synchronized_probe_requires_callback(self):
        with self.assertRaises(ValueError):EncoderStageProbe126(mode='synchronized_probe')
        drains=[];p=EncoderStageProbe126(mode='synchronized_probe',drain=lambda:drains.append(1));p.enabled=True
        p.call('stage',lambda:0)
        self.assertEqual(len(drains),2)


if __name__=='__main__':unittest.main()
