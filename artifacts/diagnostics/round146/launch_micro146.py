#!/usr/bin/env python3
"""Launch the R146 true-shape microbenchmark on 8 ranks (measurement only)."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path

ROOT = Path('/tmp/h3_micro146'); LABEL = 'micro146a'


def main():
    receipt = ROOT / (LABEL + '_process.json')
    assert not receipt.exists(), 'refuse replay of an uncertain launch'
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'receipts').mkdir(exist_ok=True)
    work = ROOT / ('cann_' + LABEL); work.mkdir(exist_ok=False)
    exports = ' '.join(['OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=1',
                        'ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', f'ASCEND_WORK_PATH={work}'])
    command = ['/bin/bash', '-c',
        ('source /home/mseco/Ascend/cann-9.2.0/set_env.sh; export PYTHONPATH=/tmp/h3_micro146:$PYTHONPATH; export '
         + exports + '; exec /home/mseco/vllm910a/bin/torch_npu_run --standalone --nproc-per-node 8 /tmp/h3_micro146/microbench146.py')]
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}
    with (ROOT / (LABEL + '.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    receipt.write_text(json.dumps({'pid': process.pid, 'command': command, 'source_sha256': sources,
                                   'measurement_only': True, 'no_model_weights_loaded': True}, indent=2) + '\n')
    print(json.dumps({'started': LABEL, 'pid': process.pid}), flush=True)
    rc = process.wait()
    data = json.loads(receipt.read_text()); data['exit_code'] = rc
    receipt.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'completed_process': process.pid, 'exit_code': rc}), flush=True)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
