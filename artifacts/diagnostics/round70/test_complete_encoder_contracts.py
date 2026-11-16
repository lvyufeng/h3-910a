"""Portable complete-encoder scope, real-loader and timer contracts."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent


def functions(path):
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


class CompleteEncoderContracts(unittest.TestCase):
    def test_real_full_loader_no_zero_embedding_or_identity_substitution(self):
        text = (ROOT / 'complete_encode_tp8.py').read_text()
        self.assertIn('encoder = candidate.MiniMaxH3Qwen3VLEncoder(', text)
        self.assertIn('loaded = encoder.load_weights(tensors())', text)
        self.assertIn('assert loaded == wanted', text)
        self.assertIn('encoder._map_weight_name(key) is not None', text)
        self.assertNotIn("yield 'model.language_model.embed_tokens.weight', torch.zeros", text)
        self.assertNotIn('model.embed_tokens = nn.Identity()', text)
        self.assertNotIn("torch.load('/tmp/h3_round54_capture01/text_inputs.pt'", text)
        self.assertIn("original['pixel_values'].dtype == torch.float16", text)
        self.assertIn("original['pixel_values_videos'] is None", text)
    def test_component_wrapper_never_copies_when_clean(self):
        node = functions(ROOT / 'complete_encode_tp8.py')['measure_part']
        branch = next(item for item in node.body if isinstance(item, ast.If) and ast.unparse(item.test) == 'not PARTS_ACTIVE')
        attrs = {item.attr for statement in branch.body for item in ast.walk(statement) if isinstance(item, ast.Attribute)}
        self.assertFalse({'cpu', 'clone', 'record', 'synchronize', 'detach'} & attrs)
        self.assertNotIn('tensor_identity', {item.id for statement in branch.body for item in ast.walk(statement) if isinstance(item, ast.Name)})
        self.assertIn("CALL_OBJECTS[name + '_output'] = output", ast.unparse(branch))
    def test_complete_timer_includes_cpu_output_excludes_proof_and_start_barrier(self):
        tree = ast.parse((ROOT / 'complete_encode_tp8.py').read_text())
        timer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'once_complete')
        clock = {'wall': 0., 'pending': 0.}
        refs = {'retained': object()}
        class Output:
            def detach(self): return self
            def cpu(self): return self
            def contiguous(self): return self
        class Encoder:
            def encode_ids(self, **kwargs):
                clock['wall'] += 3.
                clock['pending'] += 5.
                # Native encode_ids contains hidden.cpu(): modeled separately
                # from proof copying, and must remain in its wall timer.
                clock['wall'] += 7.
                return Output()
        def sync():
            clock['wall'] += clock['pending']; clock['pending'] = 0.
        def barrier(): clock['wall'] += 100.
        def identity(value): clock['wall'] += 1000.; return {'hash': 'proof'}
        namespace = {'CALL_OBJECTS': refs, 'torch': SimpleNamespace(distributed=SimpleNamespace(barrier=barrier), npu=SimpleNamespace(synchronize=sync)),
                     'time': SimpleNamespace(perf_counter=lambda: clock['wall']), 'encoder': Encoder(), 'inputs': {}, 'tensor_identity': identity}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[timer], type_ignores=[])), '<actual complete timer>', 'exec'), namespace)
        output, seconds, ids = namespace['once_complete']()
        self.assertEqual(seconds, 15.)
        self.assertEqual(clock['wall'], 2115.)
        self.assertEqual(refs, {})
        self.assertIn('complete_output', ids)
    def test_complete_timer_releases_live_refs_even_on_encode_failure(self):
        tree = ast.parse((ROOT / 'complete_encode_tp8.py').read_text())
        timer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'once_complete')
        refs = {}
        def fail(**kwargs):
            refs['large_tensor'] = object()
            raise ValueError('native encode failure')
        namespace = {'CALL_OBJECTS': refs, 'torch': SimpleNamespace(distributed=SimpleNamespace(barrier=lambda: None), npu=SimpleNamespace(synchronize=lambda: None)),
                     'time': SimpleNamespace(perf_counter=lambda: 0.), 'encoder': SimpleNamespace(encode_ids=fail), 'inputs': {}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[timer], type_ignores=[])), '<actual complete timer>', 'exec'), namespace)
        with self.assertRaises(ValueError):
            namespace['once_complete']()
        self.assertEqual(refs, {})
    def test_kernel_and_projection_math_equal_previous_text_proofs(self):
        old = functions(ROOT.parent / 'round69' / 'text_cast_tp8.py')
        actual = functions(ROOT / 'complete_encode_tp8.py')
        for name in ('row_forward', 'local', 'attention_forward', 'linear_fp32', 'mlp_forward',
                     'native_statistics_norm', 'prefix_statistics_norm', 'parameters_sha', 'metric'):
            self.assertEqual(ast.dump(old[name]), ast.dump(actual[name]), name)
    def test_fixed_vision_mode_and_BF16_original_parameters_not_model_half(self):
        text = (ROOT / 'complete_encode_tp8.py').read_text()
        self.assertIn("VLLM_OMNI_910A_VISION_FP32='1'", text)
        self.assertIn("os.environ.pop('VLLM_OMNI_910A_TE_DTYPE', None)", text)
        self.assertIn('assert all(parameter.dtype == torch.bfloat16 for parameter in encoder.parameters())', text)
        self.assertNotIn('MiniMax-H3-native-fp16-audit53', text)
        self.assertIn('No fresh FP32 processor input is substituted.', text)


if __name__ == '__main__':
    unittest.main()
