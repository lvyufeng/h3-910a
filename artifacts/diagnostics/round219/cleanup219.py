#!/usr/bin/env python3
"""Clean only supplemental R219 model/edge state, preserve baseline and receipts."""
import hashlib
import json
import os
from pathlib import Path
import shutil

root=Path('/tmp/h3_gate219')
assert root.is_dir() and root.resolve()==root
scripts={b'/tmp/h3_gate219/run_gate219.py',b'run_gate219.py',b'edge_trim219.py',b'/tmp/h3_gate219/edge_trim219.py'}
found=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdecimal() or int(proc.name)==os.getpid():continue
 try:
  cmd=(proc/'cmdline').read_bytes().split(b'\0')
  if (proc/'cwd').resolve()==root and any(arg in scripts for arg in cmd):
   found.append({'pid':int(proc.name),'cmd':cmd[0].decode(errors='replace')})
 except (PermissionError,FileNotFoundError,ProcessLookupError):pass
assert not found,found
pins={'/tmp/fa_pkg/flash_attention/ops/flash_attention/flash_attention_fwd.py':'a21c52eaa65071e6d496449bf663571bc543b1cdd818fe8950b9764e1680ffdd',
      '/tmp/fa_pkg/flash_attention/ops/flash_attention/tik_ops_utils.py':'6aa8949d19d5527ae105578ca098eff9784c96af6c63d7c5f552c49c4e413900',
      '/tmp/h3_align12401/h3_replay123/transformer.py':'5d94ba09216e3f4dce121cebbf50c960b9ddf47882fb325baffbab4c47a4fe0b',
      '/tmp/h3_perf12601/native_sources/fa910a.py':'937671efa50e20ce57c94477946d89e69b6c5e82d905ba41e5d5a203508853f1'}
actual={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in pins};assert actual==pins
removed=[]
for target in list(root.rglob('*.pt')):
 assert target.resolve()==target and root in target.parents
 target.unlink();removed.append(str(target))
for target in [root/'candidate_fa_kernels',root/'edge_kernels',root/'pkg']:
 assert target.resolve()==target and target.parent==root
 if target.exists():shutil.rmtree(target);removed.append(str(target))
for target in list(root.rglob('__pycache__')):
 assert target.resolve()==target and root in target.parents
 shutil.rmtree(target);removed.append(str(target))
record={'private_model_or_test_processes':found,'original_sources_unchanged':True,
        'source_sha256':actual,'deleted_exact_paths':removed,
        'host_only_tensor_outputs_removed':True,'candidate_private_caches_removed':True,
        'candidate_model_outputs_used':0,'no_device_or_sdk_reset':True,
        'no_other_user_process_signaled':True,'no_explicit_named_shm_or_sysv_ipc_created':True,
        'distributed_groups_closed_by_runner_finally':True,'lease_release':'pending ts_release'}
path=root/'cleanup219.json';assert not path.exists();path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
