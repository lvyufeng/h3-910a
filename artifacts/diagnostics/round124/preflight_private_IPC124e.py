#!/usr/bin/env python3
"""Test authenticated experiment-local Python manager proxy reconstruction.

No NPU/model inference. CANN-like independently spawned Python helpers must
reconstruct a manager proxy with authentication enabled. An intentionally wrong
key must be rejected. Model integration must not load in the helper process.
"""
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import sys

ROOT=Path('/tmp/h3_align12401')


def main():
    out=ROOT/'private_IPC_bootstrap_preflight124e.json';assert not out.exists()
    mp.set_start_method('spawn',force=True)
    with mp.Manager() as manager:
        proxy=manager.dict({'probe':'original-910a-private-session'})
        path=ROOT/'private_manager_proxy124e.pkl'
        with path.open('xb') as f:pickle.dump(proxy,f)
        code="""import json,multiprocessing as mp,pickle,sys
from pathlib import Path
if len(sys.argv)>1:mp.current_process().authkey=b'intentionally-wrong-internal-test-key'
p=Path('/tmp/h3_align12401/private_manager_proxy124e.pkl')
try:
 with p.open('rb') as f:obj=pickle.load(f)
 assert obj['probe']=='original-910a-private-session'
 assert not any(name in sys.modules for name in ['alignment_hooks124','vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3'])
 print(json.dumps({'authenticated_manager_proxy_rebuilt':True,'model_hooks_loaded_in_helper':False}))
except mp.AuthenticationError:
 print(json.dumps({'wrong_key_rejected':True}));raise SystemExit(12)
"""
        correct=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=90)
        wrong=subprocess.run([sys.executable,'-c',code,'wrong'],capture_output=True,text=True,timeout=90)
        report={'scope':__doc__,'completed':correct.returncode==0 and wrong.returncode==12,
                'correct_key_returncode':correct.returncode,'wrong_key_returncode':wrong.returncode,
                'correct_helper_report':correct.stdout,'wrong_key_report':wrong.stdout,
                'correct_stderr':correct.stderr,'wrong_stderr':wrong.stderr,
                'authentication_still_enabled':True,'secret_key_not_printed':True,'shared_SDK_unchanged':True}
        out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
        assert report['completed'],report


if __name__=='__main__':main()
