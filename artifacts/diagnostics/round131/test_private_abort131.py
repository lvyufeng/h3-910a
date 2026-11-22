#!/usr/bin/env python3
"""CPU-only exact-capability cleanup unit: two owned children, one untouched control.

No Torch/NPU/model/SDK access. TERM and post-grace KILL use pidfd-bound signals.
The imported helper's test capability/root are changed only in this test process.
"""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import uuid
import private_abort131 as cleanup

ROOT=Path('/tmp/h3_full5013101')
output=ROOT/'private_abort_unit131.json';assert not output.exists()
assert hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')
unit_root=ROOT/'abort_unit_receipts131';assert not unit_root.exists();unit_root.mkdir(mode=0o700)
cap='r131-'+uuid.uuid4().hex;cleanup.ROOT=unit_root;cleanup.MARKER=('H3_R131_UNIT_CAP='+cap+'\0').encode()
children=[];fds={}
try:
    for index in range(3):
        env=dict(os.environ);env.pop('H3_R127_IPC_KEY_FILE',None)
        env['H3_R131_UNIT_CAP']=cap if index<2 else 'unrelated-'+cap
        code='import signal;'+('signal.signal(signal.SIGTERM,signal.SIG_IGN);' if index==1 else '')+"print('ready',flush=True);signal.pause()"
        child=subprocess.Popen([sys.executable,'-S','-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        children.append(child)
        ready,_,_=select.select([child.stdout],[],[],10);assert ready and child.stdout.readline()==b'ready\n'
        fds[child.pid]=os.pidfd_open(child.pid)
    rows=cleanup.private_rows131();assert {r['pid'] for r in rows}=={p.pid for p in children[:2]}
    altered=dict(rows[0]);altered['start_ticks']='-1';assert not cleanup.verify_row131(altered)
    result=cleanup.abort_private131();assert result['all_private_terminated']
    assert children[0].wait(timeout=10)==-signal.SIGTERM
    assert children[1].wait(timeout=10)==-signal.SIGKILL
    assert children[2].poll() is None,'The different-capability control must remain untouched'
    result={'completed':True,'Torch_NPU_or_model_run':False,'two_exact_capability_children_terminated':True,
        'TERM_and_grace_KILL_bound_to_pidfds':True,'wrong_start_tick_rejected':True,
        'different_capability_child_untouched':True,'all_signals_targeted_owned_test_children_only':True}
finally:
    for child in children:
        fd=fds.get(child.pid)
        if fd is not None:
            if child.poll() is None:
                try:signal.pidfd_send_signal(fd,signal.SIGKILL)
                except ProcessLookupError:pass
            os.close(fd)
        child.wait(timeout=10)
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
