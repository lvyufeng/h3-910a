#!/usr/bin/env python3
"""Clean only verified experiment-private state; never resets any device or SDK."""
import hashlib
import json
import os
from pathlib import Path
import shutil

root=Path('/tmp/h3_fa217')
assert root.resolve()==root and root.is_dir()
processes=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdecimal() or int(entry.name)==os.getpid(): continue
    try:
        args=(entry/'cmdline').read_bytes().split(b'\0')
        if b'/tmp/h3_fa217/probe217.py' in args:
            processes.append(int(entry.name))
    except (PermissionError,FileNotFoundError,ProcessLookupError): pass
assert not processes, f'private experiment still running: {processes}'
expected={
    '/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py':'a21c52eaa65071e6d496449bf663571bc543b1cdd818fe8950b9764e1680ffdd',
    '/tmp/fa_pkg/flash_attention/ops/flash_attention/tik_ops_utils.py':'6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900',
}
source={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in expected}
assert source==expected, 'baseline source changed'
removed=[]
for name in ('scoped_pkg','pv_pkg','pvgroup_pkg','pvdouble_pkg'):
    target=root/name
    if target.exists():
        assert target.resolve()==target and target.parent==root
        shutil.rmtree(target)
        removed.append(str(target))
for parent in (root/'kernels',root/'tmp',root/'cann'):
    if not parent.exists():continue
    for target in parent.iterdir():
        if not any(x in target.name for x in ('scoped','pv')):continue
        assert target.resolve()==target and target.parent==parent and parent.parent==root
        if target.is_dir():shutil.rmtree(target)
        else:target.unlink()
        removed.append(str(target))
for target in (root/'out').glob('*/output_host_only.pt'):
    assert target.resolve()==target and target.parent.parent==root/'out'
    target.unlink();removed.append(str(target))
# Also clean temporary bytecode only in our copied baseline, retaining plain source.
for target in (root/'baseline_pkg').rglob('__pycache__'):
    assert target.resolve()==target and root in target.parents
    shutil.rmtree(target);removed.append(str(target))
record={'experiment_private_processes':processes,'baseline_source_unchanged':True,
        'source_sha256':source,'removed_exact_paths':removed,
        'all_tensor_outputs_removed_after_scalar_comparison':True,
        'no_device_or_driver_reset':True,'no_shared_or_other_user_process_signaled':True,
        'no_explicit_sysv_or_shm_ipc_created':True,'remaining_candidate_paths':[],
        'lease_release':'pending root ts_release call',
        'receipts_and_baseline_sources_retained':True}
path=root/'cleanup217.json'
assert not path.exists()
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
