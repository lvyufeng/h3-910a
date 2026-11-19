"""Source-only contracts for original CPU Half conditioning; no torch import/execution."""
from pathlib import Path
import ast
import importlib.util
import json
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parent
ROUND=ROOT.parent
SPEC=importlib.util.spec_from_file_location('cpu_conditioning_fixed111', ROOT/'cpu_conditioning_payload111.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MappingContracts111(unittest.TestCase):
    def test_module_import_is_stdlib_only_and_never_loads_torch(self):
        self.assertNotIn('torch',sys.modules)
        tree=ast.parse((ROOT/'cpu_conditioning_payload111.py').read_text())
        allowed={'__future__','argparse','functools','hashlib','json','os','sys','time','pathlib'}
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertTrue(all(x.name.split('.')[0]in allowed for x in node.names))
            if isinstance(node,ast.ImportFrom):self.assertIn(node.module.split('.')[0],allowed)
    def test_qkv_reorder_bijection_and_head_segment_order(self):
        order=M.qkv_reorder_order();self.assertEqual(len(order),21504)
        self.assertEqual(sorted(order),list(range(21504)))
        self.assertEqual(order[:128],tuple(range(128)))
        self.assertEqual(order[128:256],tuple(range(384,512)))
        self.assertEqual(order[7168:7296],tuple(range(128,256)))
        self.assertEqual(order[14336:14464],tuple(range(256,384)))
    def test_19_actual_Half_parameters_and_metadata_shape_gates(self):
        pins=M.load_expected_parameters(ROUND/'expected_half_parameters111.json')
        self.assertEqual(len(pins['parameters']),19)
        self.assertEqual(M.PARAMETER_SHAPES['token_refiner.blocks.0.attn.qkv_proj.weight'],(21504,5376))
        self.assertEqual(M.PARAMETER_SHAPES['token_refiner.blocks.1.mlp.fc1.weight'],(28672,5376))
        r=M.check_checkpoint_metadata(ROUND/'checkpoint_conditioning111.json');self.assertEqual(len(r['roots']),2)
    def test_portable_module_path_and_explicit_original_R78_pre_final_norm(self):
        s=(ROOT/'cpu_conditioning_payload111.py').read_text()
        self.assertNotIn('resolve().parents[4]',s)
        self.assertIn("tensor = payload[key]",s)
        self.assertIn("key not in ('pre_final_norm', 'selected50')",s)
        cases=M.load_expected_inputs(ROUND/'expected_payload_inputs111.json')
        self.assertEqual([c['tensor_key']for c in cases],['pre_final_norm','selected50','selected50'])
        self.assertIn("info['sha256'] != case['tensor_sha256']",s)

    def test_real_uppercase_nested_input_map_is_accepted(self):
        cases=M.load_expected_inputs(ROUND/'expected_inputs111.json')
        self.assertEqual(len(cases),3);self.assertEqual(cases[0]['name'],'CPU_original')
        self.assertEqual(cases[0]['tensor_sha256'],'82944620090359e7fc7eccb86471117e010b7838da83de5a43ceb46bf3453bd5')
        self.assertEqual(cases[1]['file_sha256'],'121c958b94683b11ddb8e6a2f4eea5e108366ac3ff306eb8398eddc659633e39')
    def test_input_shape_requires_single_batch_and_text_width(self):
        self.assertEqual(M.text_rows_from_shape([1,2043,5120]),(2043,5120))
        self.assertEqual(M.text_rows_from_shape([2043,5120]),(2043,5120))
        for shape in([2,2043,5120],[2043,5376],[1,0,5120]):
            with self.assertRaises(ValueError):M.text_rows_from_shape(shape)
    def test_case_path_injection_is_rejected_before_serialization(self):
        payload={'inputs':[{'name':'../escape','path':'/tmp/a.pt','tensor_sha256':'0'*64}]}
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'pins.json';p.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):M.load_expected_inputs(p)


class ReferenceNumericsContracts111(unittest.TestCase):
    def test_linear_inputs_weights_bias_remainHalf(self):
        s=(ROOT/'cpu_conditioning_payload111.py').read_text()
        self.assertIn('x.dtype != torch.float16 or weight.dtype != torch.float16',s)
        self.assertIn('return functional.linear(x, weight, bias)',s)
        self.assertNotIn('functional.linear(x.float()',s)
    def test_original_single_round_H3_norm_not_Qwen_two_round(self):
        s=(ROOT/'cpu_conditioning_payload111.py').read_text()
        norm=s[s.index('    def rms_norm('):s.index('    @staticmethod\n    def linear')]
        self.assertIn('variance = upcast.pow(2).mean(-1, keepdim=True)',norm)
        self.assertIn('out = weight.to(torch.float32) * out',norm)
        self.assertEqual(norm.count('.to(input_dtype)'),1)
    def test_unmasked_noncausal_MATH_Half_attention_no_RoPE(self):
        s=(ROOT/'cpu_conditioning_payload111.py').read_text()
        self.assertIn('with math_sdpa_context(torch):',s)
        self.assertIn('attn_mask=None',s);self.assertIn('is_causal=False',s)
        self.assertIn('scale=SOFTMAX_SCALE',s)
        self.assertIn("gate, up = gate_up.chunk(2, dim=-1)",s)
        self.assertIn('activated = functional.silu(gate) * up',s)
    def test_exact_both_passes_parameters_inputs_source_gates(self):
        s=(ROOT/'cpu_conditioning_payload111.py').read_text()
        self.assertIn('passes < 2',s)
        self.assertIn('if infos != first_infos:',s)
        self.assertIn("parameter_sha_after == parameter_sha_before",s)
        self.assertIn("'full_encoder_precision_accepted': False",s)
        self.assertIn("'whole_goal_complete': False",s)
        self.assertIn("'numerical_tolerance': None",s)
    def test_four_required_stages_and_bounded_saved_bytes(self):
        self.assertEqual(M.MANDATORY_STAGES,('projected','refiner0','refiner1','finalnorm'))
        self.assertEqual(M.expected_stage_shape('refiner0',2043),(2043,5376))
        self.assertEqual(4*2043*5376*2,87865344)
        self.assertLess(4*2043*5376*2,M.MAX_STAGE_FILE_BYTES)


class SafeOutputContracts111(unittest.TestCase):
    def test_publish_is_exclusive_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d).resolve()/'receipt.json';M.write_json_atomic(path,{'completed':True})
            self.assertEqual(json.loads(path.read_text()),{'completed':True})
            with self.assertRaises(FileExistsError):M.write_json_atomic(path,{'bad':True})
            self.assertFalse(path.with_name(path.name+'.tmp').exists())
    def test_reused_reference_failure_or_tensor_root_is_refused(self):
        for name in(M.REFERENCE_JSON,M.FAILURE_JSON,M.STAGE_TENSOR_TEMPLATE%'CPU_original'):
            with tempfile.TemporaryDirectory()as d:
                (Path(d)/name).write_text('preserve')
                with self.assertRaises(FileExistsError):M.refuse_existing_outputs(Path(d),['CPU_original'])
                self.assertEqual((Path(d)/name).read_text(),'preserve')
    def test_symlink_destination_and_reused_temporary_are_refused(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);original=root/'gold.json';original.write_text('unchanged')
            link=root/'link.json';link.symlink_to(original)
            with self.assertRaises(FileExistsError):M.write_json_atomic(link,{'bad':True})
            (root/'new.json.tmp').write_text('keep')
            with self.assertRaises(FileExistsError):M.write_json_atomic(root/'new.json',{'bad':True})
            self.assertEqual(original.read_text(),'unchanged')


if __name__=='__main__':unittest.main()
