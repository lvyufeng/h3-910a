#!/usr/bin/env python3
"""Consolidate actual primitive, storage, and measurement-protocol checks."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
cpu=json.loads((root/'cpu_primitives.json').read_text())
assert cpu['passed']
rows=[json.loads((root/f'hccl_rank{i}.json').read_text()) for i in range(8)]
assert all(r['passed'] and r['hccl_contiguous_mutates_original'] and r['hccl_noncontiguous_requires_return'] for r in rows)
timer=json.loads((root/'timer_test.json').read_text())
assert timer['passed'] and timer['perf_mode_explicit_measurement_sync'] and timer['perf_disabled_original_cancellation_behavior']
store=json.loads((root/'native_storage_conversion.json').read_text())
bits=json.loads((root/'native_storage_bitcheck.json').read_text())
small=json.loads((root/'storage_test.json').read_text())
assert store['complete'] and bits['passed'] and bits['tensors_checked']==535 and small['passed']
summary={'host_contract_tests_passed':9,'hccl_ranks_verified':8,
         'old_and_new_contiguous_primitives_exact_small_cpu_reference':True,
         'return_only_mock_and_noncontiguous_real_hccl_need_return_value':True,
         'collective_bug_not_established_as_previous_video_cause':True,
         'measurement_protocol':'PERF=1 explicit torch.npu.synchronize through platform; PERF=0 asynchronous unchanged',
         'native_storage_shards':13,'native_storage_bytes':store['total_bytes'],
         'native_storage_raw_bit_tensors_verified':535,
         'bf16_roundtrip_changed_elements':store['total_bf16_roundtrip_changed_elements'],
         'conversion_not_bf16_lossless':True,
         'qualification':'Primitive/metadata/storage tests; full Qwen and official GPU-reference quality equivalence are not established.'}
out=root/'validation_summary.json'
if out.exists(): raise RuntimeError('Refuse existing summary')
out.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
