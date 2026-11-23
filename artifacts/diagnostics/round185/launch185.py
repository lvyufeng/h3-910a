#!/usr/bin/env python3
"""R185 launcher: run native then fast, both 8-rank, and summarise the step times."""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path('/tmp/h3_fast185')
R = '/home/mseco/Ascend/cann-9.2.0'
STEPS = os.environ.get('H3_FAST185_STEPS', '3')

BASE_ENV = {
    'VLLM_OMNI_DLO_DP_WAVE_TIMEOUT': '7200',
    'VLLM_OMNI_910A_FLASH_ATTN': '1',
    'VLLM_OMNI_910A_FP16_SEQ': '1',
    'VLLM_OMNI_910A_TE_DTYPE': 'fp16',
    'VLLM_OMNI_910A_TEXT_HF_MATH': '1',
    'VLLM_OMNI_910A_NO_FUSED_ACT': '1',
    'VLLM_OMNI_910A_NO_FUSED_NORM': '1',
    'H3_NPU_NAN_COMPAT124': '1',
    'H3_NPU_VAE_PRESERVE_FP32_PARAMS124': '1',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
    'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
    'LD_LIBRARY_PATH': f'{R}/opp/vendors/customize/op_api/lib:{R}/lib64',
    'H3_FAST185_STEPS': STEPS,
}


def run(label, audit_dir, fast_flag):
    out = ROOT / 'out' / label
    out.mkdir(parents=True, exist_ok=True)
    # the benchmark asserts its output directory exists before torchrun starts, and it resolves
    # that path as ROOT/<label>, so the launcher has to create exactly that
    (ROOT / label).mkdir(parents=False, exist_ok=True)
    # start from the real environment: replacing it wholesale drops HOME, which makes
    # GEInitialize fail with an ACL error that looks nothing like a missing variable
    env = os.environ.copy()
    env.update(BASE_ENV)
    env['H3_FAST185_LABEL'] = label
    # the model source tree is pinned for this replay, exactly as R149 did; the audit dir goes
    # first so the native run resolves the untouched matrix_safety126
    env['PYTHONPATH'] = f'{audit_dir}:{ROOT}:/tmp/h3_align12401'
    env['ASCEND_WORK_PATH'] = str(ROOT / ('cann_' + label))
    if fast_flag:
        env['H3_FAST_AUDIT'] = '1'
    else:
        env.pop('H3_FAST_AUDIT', None)
    cmd = ('source ' + R + '/set_env.sh; exec /home/mseco/vllm910a/bin/torch_npu_run '
           '--standalone --nproc-per-node 8 ' + str(ROOT / 'run185.py'))
    log = out / (label + '.log')
    print(f'[launch] {label} audit={audit_dir} fast={fast_flag}', flush=True)
    with log.open('xb') as fh:
        rc = subprocess.call(['/bin/bash', '-c', cmd], cwd=str(ROOT), env=env,
                             stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT)
    print(f'[done] {label} exit={rc}', flush=True)
    return rc, log


def summarise(label, log):
    txt = log.read_text(errors='replace') if log.exists() else ''
    steps = [float(m) for m in re.findall(r'step[_\s]*time[=:\s]+([0-9.]+)', txt, re.I)]
    medians = [float(m) for m in re.findall(r'median[=:\s]+([0-9.]+)', txt, re.I)]
    return {'label': label, 'step_times': steps[:10], 'medians': medians[:5],
            'log_tail': txt[-600:]}


def main():
    rc_n, log_n = run('native185', str(ROOT / 'native'), False)
    rc_f, log_f = run('fast185', str(ROOT / 'fast'), True)
    summary = {'steps': STEPS, 'native_exit': rc_n, 'fast_exit': rc_f,
               'native': summarise('native185', log_n), 'fast': summarise('fast185', log_f)}
    (ROOT / 'out' / 'summary185.json').write_text(json.dumps(summary, indent=2) + '\n')
    import hashlib
    pins = {}
    for path in ('/tmp/h3_align12401/h3_replay123/transformer.py',
                 '/tmp/h3_align12401/NPU_encoder_FP16_124.py',
                 str(ROOT / 'native' / 'matrix_safety126.py'),
                 str(ROOT / 'fast' / 'matrix_safety126.py')):
        try:
            pins[path] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        except OSError as exc:
            pins[path] = f'missing: {exc}'
    summary['source_sha256'] = pins
    (ROOT / 'out' / 'summary185.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('steps', 'native_exit', 'fast_exit')}, indent=1))
    for k in ('native', 'fast'):
        print(k, 'step_times=', summary[k]['step_times'], 'medians=', summary[k]['medians'])


if __name__ == '__main__':
    main()
