"""Pin baseline-search evidence and prevent BF16 acceptance/false FP16-gold claims."""
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent


class FP16BaselineContracts(unittest.TestCase):
    def test_public_search_is_bounded_and_fp16_run_candidates_exist(self):
        r=json.loads((ROOT/'search_receipts77.json').read_text())
        self.assertTrue(r['search_is_bounded_not_exhaustive']);self.assertTrue(r['no_model_weights_downloaded']);self.assertTrue(r['no_cpu_or_npu_inference_launched'])
        candidates={item['url']for item in r['queries']['gh_repository_fp16_search']['items']}
        self.assertIn('https://github.com/Amduraznak/minimax-h3-fp16-fix',candidates)
        self.assertIn('https://github.com/Icbears/minimax-h3-v100-patch',candidates)
        for source in r['queries'].values():
            if 'path'in source:self.assertEqual(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest(),source['sha256'])
    def test_pinned_ports_are_not_executed_and_not_cpu_gold(self):
        r=json.loads((ROOT/'port_source_receipts77.json').read_text());selection=json.loads((ROOT/'baseline_selection77.json').read_text())
        for source in r['sources'].values():
            self.assertEqual(source['status'],200);self.assertTrue(source['source_only_not_executed'])
            self.assertEqual(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest(),source['sha256'])
        self.assertTrue(selection['public_FP16_run_implementations_found']);self.assertFalse(selection['public_full_stack_CPU_FP16_gold_found_in_checked_scope'])
        self.assertTrue(selection['no_global_nonexistence_claim'])
        for candidate in selection['candidates']:self.assertFalse(candidate['CPU_FP16_golden_tensor_or_receipt'])
    def test_header_dtype_evidence_is_header_only_and_exactly_pinned(self):
        r=json.loads((ROOT/'header_receipts77.json').read_text());self.assertTrue(r['bounded_no_tensor_payloads'])
        counts={'official_dit_first':{'F32':9,'BF16':45},'official_text_first':{'BF16':43},'comfy_unpruned_dit':{'F32':13,'BF16':522},'comfy_unpruned_text':{'BF16':902},'comfy_video_vae_fp16':{'F16':562}}
        for name,source in r['targets'].items():
            self.assertNotIn('failure',source);self.assertFalse(source['weight_payload_downloaded']);self.assertLessEqual(source['header_bytes'],262144)
            self.assertTrue(all(p['status']==206 for p in source['responses']))
            p=ROOT/source['header_path'];self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),source['saved_header_file_sha256'])
            header=json.loads(p.read_text());self.assertEqual(dict(collections.Counter(v['dtype']for k,v in header.items()if k!='__metadata__')),counts[name])
            self.assertEqual(source['dtype_counts'],counts[name])
    def test_selected_reference_is_official_fp16_recompute_not_cast_BF16_output(self):
        r=json.loads((ROOT/'acceptance_fp16_contract77.json').read_text())
        self.assertFalse(r['BF16_cross_precision_agreement_required']);self.assertTrue(r['old_R74_BF16_discrepancy_is_not_a_blocking_acceptance_gate'])
        self.assertTrue(r['old_BF16_outputs_cast_to_FP16_are_not_an_FP16_reference']);self.assertTrue(r['reference_must_be_recomputed'])
        self.assertFalse(r['CPU_FP16_reference_generated']);self.assertFalse(r['NPU_FP16_precision_accepted']);self.assertFalse(r['whole_performance_goal_complete'])
        self.assertTrue(r['selected_base_source']['official_CFG_distilled_base_is_allowed']);self.assertFalse(r['selected_base_source']['additional_few_step_Turbo_or_FlashGen_LoRA'])
    def test_official_base_has_no_added4step_schedule(self):
        index=json.loads((ROOT/'search_sources'/'hf_official_base_index.txt').read_text());self.assertNotIn('base_schedule',index['_minimax_h3'])
        self.assertEqual(index['_minimax_h3']['sigma_shift_scales'],{'video':12.0,'audio':3.0})
        self.assertEqual(index['_minimax_h3']['tasks'],['t2va','fl2va'])
    def test_actual_installed_dtype_symbol_not_upstream_bf16_assumption(self):
        r=json.loads((ROOT/'current_runtime_dtype_proof77.json').read_text());source=ROOT.parent/'round61'/'source'/'minimax_h3_transformer.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),r['actual_installed_source_hash_proven_by_round76_final_verification'])
        self.assertIn(r['constant_expression'],source.read_text());self.assertEqual(r['default_constant_value'],'torch.float16')
        self.assertTrue(r['online_main_branch_must_not_be_substituted_for_installed_code']);self.assertTrue(r['not_a_new_dtype_tensor_trace'])
    def test_power_of_two_linear_claim_needs_bias_and_range_conditions(self):
        x,w,b,k=1.,2.,1.,64.
        y=w*x+b;prescaled=(w*(x/k)+b)*k
        self.assertNotEqual(y,prescaled);self.assertEqual(prescaled-y,(k-1)*b)
        self.assertEqual((w*(x/k))*k,w*x)
    def test_RMSNorm_scaling_changes_epsilon_effect(self):
        q,epsilon,s=1.,1e-5,256.
        native=q/math.sqrt(q*q+epsilon)
        scaled=(q/s)/math.sqrt((q/s)**2+epsilon)
        self.assertGreater(abs(native-scaled),.1)
    def test_historical_BF16_isolation_script_is_not_selected_as_new_gold(self):
        r=json.loads((ROOT/'acceptance_fp16_contract77.json').read_text());self.assertIn('NOTexecuted',r['abandoned_before_execution'])
        self.assertTrue(r['no_NPU_or_CPU_inference_launched_this_search_stage']);self.assertTrue(r['ts217_lease_released_after_user_requested_search_first'])


if __name__=='__main__':unittest.main()
