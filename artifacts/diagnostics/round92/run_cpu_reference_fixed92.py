#!/usr/bin/env python3
"""Invoke the immutable R92 CPU oracle with explicit metric callables.

The first CLI call reached reference_gate but failed when its optional local
``tensor_info=None`` shadowed the module function. Original math and reference
SHA gates are unchanged; the new root supplies the already reviewed R84 helper.
"""
from pathlib import Path
import json
import hashlib
ROOT=Path('/tmp/h3_round92_case02')


def main():
    import original_cpu_block25_92 as original
    import full_row_fp32_fixed84 as metrics
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()=='838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b'
    driver={'scope':__doc__,'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_oracle_SHA256':'838027c883bed6d1e5a0b839e0c2accde2457bef9cc81d58129fdcad38d65f4b',
            'metric_callables_explicit_not_None':True,'original_CPU_reference_gate_and_math_not_changed':True,
            'no_new_tolerance_or_DeepStack_finalnorm':True}
    with(ROOT/'cpu_driver_fixed92.json').open('x')as f:f.write(json.dumps(driver,indent=2)+'\n')
    receipt=original.run_reference(root=ROOT,compare=metrics.compare,tensor_info=metrics.tensor_info)
    print(json.dumps({'completed':receipt['completed'],'reference_gate':receipt['reference_gate'],
                      'input_error_amplification':receipt['input_error_amplification'],
                      'outputs_file_SHA256':receipt['outputs_file_SHA256']},indent=2),flush=True)


if __name__=='__main__':main()
