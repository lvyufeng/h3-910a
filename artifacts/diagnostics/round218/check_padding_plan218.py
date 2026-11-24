#!/usr/bin/env python3
"""Contracts for exact padding-only scheduling; no causal/sparse token pruning."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
cases=[]
for source_n,kernel_n,br,bc,pad in [(41792,43776,128,304,23),
                                   (2048,2048,128,256,257),
                                   (2048,2048,128,256,256),
                                   (4096,4096,128,256,23)]:
    valid=source_n-pad
    tr=math.ceil(source_n/br);tc=math.ceil(valid/bc)
    scheduled_q_rows=tr*br;scheduled_key_cols=tc*bc
    assert source_n<=scheduled_q_rows<=kernel_n
    assert valid<=scheduled_key_cols<=kernel_n
    # Every source query and valid key, including partial final tiles, is retained.
    assert all(i//br<tr for i in range(source_n))
    assert all(j//bc<tc for j in range(valid))
    assert all(j>=valid for j in range(scheduled_key_cols,kernel_n))
    cases.append({'input_N':source_n,'kernel_N':kernel_n,'Br':br,'Bc':bc,'original_pad':pad,
                  'valid_keys':valid,'Tr':tr,'Tc':tc,
                  'all_input_query_rows_scheduled':True,'all_valid_keys_scheduled':True,
                  'only_fully_invalid_key_tiles_removed':True,
                  'query_rows_beyond_input_N_only_removed':True,
                  'tile_work_ratio_to_padded_baseline':tr*tc/((kernel_n//br)*(kernel_n//bc)),
                  'not_numerical_equivalence_proof':True})
(HERE/'padding_plan_contracts218.json').write_text(json.dumps({'cases':cases,'pass':True},indent=2)+'\n')
print(json.dumps(cases,indent=2))
