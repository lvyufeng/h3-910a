#!/usr/bin/env python3
"""Read-only bounded compressed R128 receipts/audits, no weights or activations."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

SCRIPT=r'''
from pathlib import Path
import io,json,os,select,sys,tarfile
root=Path('/tmp/h3_overlap12801');session=root/'engine_overlap128a';receipt=root/'engine_overlap128a_process.json'
row=json.loads(receipt.read_text());assert row['pid']==int(sys.argv[1])
if row.get('exit_code') is None:
    parent=row['launcher_pid'];proc=Path('/proc',str(parent))
    if proc.exists():
        assert b'launch_lifecycle127.py' in (proc/'cmdline').read_bytes()
        fd=os.pidfd_open(parent)
        try:
            ready,_,_=select.select([fd],[],[],7500);assert ready,'Private launcher wait timed out'
        finally:os.close(fd)
    row=json.loads(receipt.read_text());assert row.get('exit_code') is not None,'Final atomic receipt is missing'
files=[receipt,session/'lifecycle_result127.json',root/'launch_source_pins128a_final.json']
files += sorted(root.glob('formal_replay_rank*128.json'))+sorted(root.glob('cohort_failure*.json'))
files += sorted(session.glob('worker_runtime_pid*127.jsonl'))+sorted(session.glob('initialization_pack_pid*127.jsonl'))
files += sorted(session.glob('initialization_crash_pid*127.txt'))+sorted(session.glob('r*/audit_complete_pid*127.json'))
assert all(p.is_file() and p.suffix in ('.json','.jsonl','.txt') and not p.is_symlink() for p in files)
assert sum(p.stat().st_size for p in files)<=32*1024**2
output=io.BytesIO()
with tarfile.open(fileobj=output,mode='w:gz',compresslevel=6) as tar:
    for path in files:tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
sys.stdout.buffer.write(output.getvalue())
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();assert not args.output_dir.exists()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=20',
        '-o','ControlMaster=no','-o','ControlPath=none','ts-217',f'/usr/bin/python3 - {args.pid}']
    proc=subprocess.run(command,input=SCRIPT.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if proc.returncode:raise RuntimeError(proc.stderr.decode(errors='replace'))
    assert len(proc.stdout)<=32*1024**2
    args.output_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(proc.stdout),mode='r:gz') as tar:
        members=tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).suffix in ('.json','.jsonl','.txt') for m in members)
        assert sum(m.size for m in members)<=32*1024**2
        tar.extractall(args.output_dir,filter='data')
    process=json.loads((args.output_dir/'engine_overlap128a_process.json').read_text())
    result=json.loads((args.output_dir/'engine_overlap128a/lifecycle_result127.json').read_text())
    report={'process_exit_code':process['exit_code'],'native_result_completed':result['completed'],'full49_requests':len(result['requests']),
        'candidate_declined':result.get('candidate_49_generation_not_launched'),'failure':result.get('failure'),
        'compressed_small_evidence_bytes':len(proc.stdout),'weights_or_activation_archives_transferred':False}
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
