#!/usr/bin/env python3
"""Clean only R218 private state after collection and comparison; no device reset."""
from pathlib import Path
import os
import hashlib
import json
import shutil

roots=[Path('/tmp/h3_path218'),Path('/tmp/h3_broadcast218'),Path('/tmp/h3_trim218'),Path('/tmp/h3_trim_model218')]
assert all(r.is_dir() and r.resolve()==r for r in roots)
known_scripts={b'run218.py',b'/tmp/h3_path218/run218.py',b'run_trim_model218.py',b'/tmp/h3_trim_model218/run_trim_model218.py',
               b'fa_probe218.py',b'/tmp/h3_broadcast218/fa_probe218.py',b'trim_probe218.py',b'/tmp/h3_trim218/trim_probe218.py',
               b'broadcast_unit218.py',b'boundary_trim218.py'}
found=[]
for p in Path('/proc').iterdir():
    if not p.name.isdecimal() or int(p.name)==os.getpid():continue
    try:
        cmd=(p/'cmdline').read_bytes().split(b'\0')
        if not cmd:continue
        cwd=(p/'cwd').resolve()
        if cwd in roots and any(a in known_scripts for a in cmd):found.append({'pid':int(p.name),'cwd':str(cwd)})
    except (PermissionError,FileNotFoundError,ProcessLookupError):pass
assert not found,found
pins={
 '/tmp/fa_pkg/flash_attention/ops/flash_attention/tik_ops_utils.py':'6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900',
 '/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py':'a21c52eaa65071e6d496449bf663571bc543b1cdd818fe8950b9764e1680ffdd',
 '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
 '/tmp/h3_perf12601/native_sources/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'}
actual={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in pins}
assert actual==pins,'original source changed'
removed=[]
# No activation or output tensor is uploaded; remove host-only comparison outputs now.
for root in roots:
    for file in list(root.rglob('*.pt')):
        assert file.resolve()==file and root in file.parents
        file.unlink();removed.append(str(file))
# Remove isolated candidate packages and cached executable kernels, retaining scalar receipts.
for target in [Path('/tmp/h3_broadcast218/candidate_pkg'),
               Path('/tmp/h3_broadcast218/kernels/small_candidate'),
               Path('/tmp/h3_broadcast218/kernels/real_candidate'),
               Path('/tmp/h3_trim_model218/candidate_fa_kernels'),
               Path('/tmp/h3_trim218/pkg')]:
    assert target.resolve()==target and any(root in target.parents for root in roots)
    if target.exists():shutil.rmtree(target);removed.append(str(target))
for parent in [Path('/tmp/h3_trim218/kernels'),Path('/tmp/h3_trim218/tmp'),Path('/tmp/h3_trim218/cann')]:
    if parent.exists():
        for target in parent.iterdir():
            if target.name.startswith('trim_') or target.name=='boundary_trim':
                assert target.resolve()==target and target.parent==parent
                if target.is_dir():shutil.rmtree(target)
                else:target.unlink()
                removed.append(str(target))
for root in roots:
    for target in list(root.rglob('__pycache__')):
        assert target.resolve()==target and root in target.parents
        shutil.rmtree(target);removed.append(str(target))
record={'private_model_or_FA_processes':found,'source_sha256':actual,'source_unchanged':True,
        'deleted_exact_private_paths':removed,'no_device_driver_or_sdk_reset':True,
        'no_other_user_process_signaled':True,'all_host_only_tensor_outputs_removed':True,
        'private_candidate_binary_caches_removed':True,'kernel_defaults_unmodified':True,
        'no_explicit_sysv_or_named_shm_created':True,'distributed_groups_closed_by_runner_finally':True,
        'lease_release':'pending ts_release','receipts_retained_on_host':True}
path=roots[0]/'cleanup218.json';assert not path.exists()
path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
