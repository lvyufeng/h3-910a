"""Source and actual-data gates for output-column and genuine raw-transport probes."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from column_plan75 import WIDTHS,indices,padded_width
from summarize_capture75 import summarize as capture_summary
from summarize_column75 import summarize as column_summary
import summarize_column75 as column_gate
from summarize_wire75 import summarize,summarize_mode

ROOT=Path(__file__).resolve().parent


class SourceContracts(unittest.TestCase):
    def test_full_output_partition_has_no_holes_overlap_or_splitK(self):
        for name,width in WIDTHS.items():
            parts=[indices(name,rank)for rank in range(8)]
            self.assertEqual(sorted(x for part in parts for x in part),list(range(width)))
            self.assertTrue(all(len(part)==width//8 for part in parts))
        self.assertEqual(padded_width('fc1'),538);self.assertEqual(padded_width('fc1',True),544)
    def test_qkv_head_groups_are_not_simple_contiguous432_rows(self):
        for rank in range(8):
            self.assertEqual(indices('qkv',rank),[group*1152+i for group in range(3)for i in range(rank*144,(rank+1)*144)])
        self.assertNotEqual(indices('qkv',0),list(range(432)))
    def test_raw_wire_is_view_not_numeric_to_and_exact_nd_guard(self):
        text=(ROOT/'raw_wire75.py').read_text();tree=ast.parse(text)
        attrs={node.func.attr for node in ast.walk(tree)if isinstance(node,ast.Call)and isinstance(node.func,ast.Attribute)}
        self.assertNotIn('to',attrs);self.assertIn('view',attrs)
        self.assertIn('value.shape[-1]%2',text);self.assertIn('value.storage_offset()%2',text)
        self.assertIn('value.data_ptr()%32',text);self.assertIn('not in(0,2)',text)
        self.assertIn('torch.npu.current_device()',text)
    def test_safe_probe_allows_only_known_numeric_enum_after_own_filehash(self):
        text=(ROOT/'column_probe_safe75.py').read_text()
        self.assertLess(text.index("hashlib.sha256(file.read_bytes()).hexdigest()==meta['file_sha256']"),text.index('with torch.serialization.safe_globals([Format])'))
        self.assertIn('payload=torch.load(file,map_location=\'cpu\',weights_only=True)',text)
        self.assertNotIn('weights_only=False)',text)
    def test_padding_recovery_keeps_originalK_biasfusion_and_no_precision_toggle(self):
        text=(ROOT/'fc2_padding75.py').read_text()
        self.assertIn('F.linear(x,wp,bp)',text);self.assertIn('for width in(160,192,256,512)',text)
        self.assertIn("payload['w'].index_select(0,idxcpu)",text)
        self.assertNotIn('allow_hf32=',text);self.assertNotIn('deterministic_algorithms',text)
        self.assertIn('wp[valid:]==0',text);self.assertIn('bp[valid:]==0',text)
    def test_experimental_group_all_gather_is_not_reduce_fp16_path(self):
        source=ROOT.parent/'round59'/'source'/'group_coordinator.py'
        tree=ast.parse(source.read_text());group=next(node for node in tree.body if isinstance(node,ast.ClassDef)and node.name=='GroupCoordinator')
        gather=next(node for node in group.body if isinstance(node,ast.FunctionDef)and node.name=='all_gather')
        reduce=next(node for node in group.body if isinstance(node,ast.FunctionDef)and node.name=='all_reduce')
        self.assertNotIn('torch.float16',ast.unparse(gather));self.assertIn('torch.float16',ast.unparse(reduce))
        wire=json.loads((ROOT/'wire_summary.json').read_text())
        self.assertEqual(wire['modes']['bf16']['group_source_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())


class ActualEvidence(unittest.TestCase):
    def test_current_full_encoder_capture_and_all_real_operands_unchanged(self):
        r=capture_summary();self.assertTrue(r['complete_output_identity_accepted']);self.assertEqual(r['complete_outputs_verified'],8)
        self.assertEqual(r['actual_linear_fixtures'],4);self.assertEqual(r['total_fixture_bytes'],317946420)
        self.assertFalse(r['whole_encoder_acceleration_proven']);self.assertFalse(r['official_goal_complete'])
    def test_retained_fc2_144_failure_not_reclassified_by_zero_padding_success(self):
        r=column_summary();self.assertFalse(r['unmodified_144_fc2_family_accepted'])
        self.assertEqual(r['rejected_fc2_144']['BF16_vs_full']['different_words'],354)
        self.assertTrue(r['rejected_fc2_144']['acceptance_rejected'])
        self.assertEqual(set(r['separate_fc2_zero_padding']),{'160','192','256','512'})
        self.assertEqual(r['smallest_tested_strictly_exact_fc2_width'],160)
        self.assertFalse(r['all_later_layers_and_mergers_covered']);self.assertFalse(r['whole_encoder_acceleration_proven'])
    def test_padding_recovery_from_different_fixture_cannot_repair_retained_failure(self):
        failure=json.loads((ROOT/'column_determinism_failure02.json').read_text())
        padding=json.loads((ROOT/'fc2_padding01.json').read_text());padding['fixture_sha256']='f'*64
        with patch('summarize_column75.json.loads',side_effect=[failure,padding]),self.assertRaisesRegex(ValueError,'different FC2 fixture'):column_summary()
    def test_false_mode_label_cannot_override_actual_gemm_width(self):
        failure=json.loads((ROOT/'column_determinism_failure02.json').read_text())
        failure['cases'][2]['modes']['zero_pad_999']=failure['cases'][2]['modes'].pop('zero_pad_544')
        with patch('summarize_column75.json.loads',side_effect=[failure]),self.assertRaisesRegex(ValueError,'Mode labels'):column_summary()
    def test_true8rank_directBF16_and_INT32_word_transports_have_same_source_scope(self):
        r=summarize();self.assertTrue(r['direct_bf16_gather_bits_proven_for_current_experimental_encoder_group'])
        self.assertTrue(r['int32_raw_pair_gather_bits_proven']);self.assertFalse(r['integer_codec_required_for_this_direct_gather'])
        for mode in('bf16','int32'):
            self.assertEqual(r['modes'][mode]['gather_proofs_verified'],384)
            self.assertEqual(r['modes'][mode]['raw_view_roundtrips'],48)
            self.assertEqual(r['modes'][mode]['guard_rejections_verified'],24)
        self.assertFalse(r['production_communicator_proven']);self.assertFalse(r['whole_encoder_acceleration_proven'])
    def test_word_proofs_preserve_signedzero_all_finite_and_exceptional_payloads(self):
        for mode in('int32','bf16'):
            rows=[json.loads((ROOT/('wire_'+mode)/f'wire_rank{i}.json').read_text())for i in range(8)]
            for row in rows:
                for case in row['cases']:
                    self.assertEqual(case['distinct_bf16_words_in_first65568_source_words'],65536)
                    self.assertTrue(case['source_readonly'])
                    for v in case['directions'].values():
                        for p in[v['proof'],*v['repeat_proofs']]:self.assertEqual(p['different_bf16_words'],0)
    def test_odd_int32_width_is_valid_because_it_decodes_to_even_bf16_width(self):
        row=json.loads((ROOT/'wire_int32'/'wire_rank0.json').read_text())
        case=next(value for value in row['cases']if(value['rows'],value['bf16_columns'])==(8064,538))
        self.assertEqual(case['raw_device_view_roundtrip']['shape_int32'],[8064,269])
        self.assertTrue(case['raw_device_view_roundtrip']['raw_pairs_equal'])
    def test_cpu_backend_mode_mix_and_bad_word_cannot_pass_transport_gate(self):
        original=[json.loads((ROOT/'wire_int32'/f'wire_rank{i}.json').read_text())for i in range(8)]
        for mutation in('cpu_backend','mode_flag','bad_word','production_scope'):
            rows=deepcopy(original)
            if mutation=='cpu_backend':rows[0]['device_group_backend']='gloo'
            elif mutation=='mode_flag':rows[0]['integer_transport_only']=False
            elif mutation=='bad_word':rows[0]['cases'][0]['directions']['0']['proof']['different_bf16_words']=1
            else:rows[0]['production_communicator_proven']=True
            with self.subTest(mutation=mutation),patch('summarize_wire75.json.loads',side_effect=rows),self.assertRaises(ValueError):summarize_mode('int32')
    def test_failed_pickle_fixture_remains_rejected_and_not_adapted_as_math_failure(self):
        r=json.loads((ROOT/'column_pickle_failure01.json').read_text())
        self.assertFalse(r.get('completed',False));self.assertEqual(r['failure']['type'],'UnpicklingError')
        self.assertEqual(r['cases'],[])
        self.assertIn('Format',r['failure']['message'])
    def test_final_sources_and_actual_fixture_fingerprints_match(self):
        r=json.loads((ROOT/'final_verification.json').read_text());expected=json.loads((ROOT/'expected_sources.json').read_text())
        self.assertTrue(r['all_match']);self.assertFalse(r['production_modified']);self.assertEqual(len(r['production']),23)
        self.assertEqual(len(r['fixtures']),5)
        for category in('production','diagnostics','fixtures'):
            self.assertEqual(set(expected[category]),set(r[category]))
            for path,digest in expected[category].items():
                self.assertTrue(r[category][path]['matches']);self.assertEqual(r[category][path]['sha256'],digest)
                local=ROOT/Path(path).name
                if Path(path).name=='candidate_padded_text_encoder.py':local=ROOT.parent/'round65'/'source'/'candidate_padded_text_encoder.py'
                if Path(path).name=='evidence_io75.py':local=ROOT.parent/'round63'/'evidence_io63.py'
                if local.is_file():self.assertEqual(hashlib.sha256(local.read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main()
