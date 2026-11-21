#!/usr/bin/env python3
"""Wait for one verified R127 process, then collect only bounded small evidence.

No recursive SCP, no weights/latents/media archive transfer, no blind replay. The
waiting uses Linux pidfd on the existing generator, not a polling/sleep loop.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    remote=r'''
import io,json,os,select,sys,tarfile
from pathlib import Path
root=Path('/tmp/h3_runtime12701');receipt=root/'engine_lifecycle127c_process.json'
meta=json.loads(receipt.read_text());pid=int(sys.argv[1]);assert meta['pid']==pid
if meta.get('exit_code') is None:
    parent=meta['launcher_pid'];proc=Path('/proc',str(parent))
    if proc.exists():
        parent_command=(proc/'cmdline').read_bytes()
        assert b'launch_lifecycle127.py' in parent_command
        fd=os.pidfd_open(parent)
        try:
            ready,_,_=select.select([fd],[],[],7200)
            assert ready,'Launcher wait timed out; do not collect an unfinished run'
        finally:os.close(fd)
    meta=json.loads(receipt.read_text())
    assert meta.get('exit_code') is not None,'Launcher ended without final atomic receipt; preserve unknown state'
session=root/'engine_lifecycle127c'
# All selected files are JSON/source receipts, never model/activation archives.
paths=[receipt]+sorted(session.glob('*.json'))+sorted(session.glob('worker_runtime_pid*127.jsonl'))+sorted(session.glob('r*/audit_complete_pid*127.json'))
assert all(p.is_file() and p.suffix in ('.json','.jsonl') for p in paths)
assert sum(p.stat().st_size for p in paths)<=20*1024**2
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for path in paths:archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
'''
    cmd=['ssh','-C','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30',
         '-o','ServerAliveCountMax=20','-o','ControlMaster=no','-o','ControlPath=none','ts-217',
         '/usr/bin/python3 - '+str(args.pid)]
    result=subprocess.run(cmd,input=remote.encode(),capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    assert len(result.stdout)<=21*1024**2
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:') as archive:
        for member in archive.getmembers():
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert path.suffix in ('.json','.jsonl') and member.size<=4*1024**2
            target=args.output_dir.joinpath(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            assert not target.exists()
            content=archive.extractfile(member).read()
            if path.suffix=='.json':json.loads(content)
            else:
                for line in content.splitlines():json.loads(line)
            target.write_bytes(content)
    process=json.loads((args.output_dir/'engine_lifecycle127c_process.json').read_text())
    session=args.output_dir/'engine_lifecycle127c'
    result_path=session/'lifecycle_result127.json'
    current=json.loads(result_path.read_text()) if result_path.exists() else {}
    print(json.dumps({'process_exit_code':process.get('exit_code'),'full_lifecycle_completed':current.get('completed',False),
        'completed_requests':len(current.get('requests',[])),'failure':current.get('failure'),
        'small_evidence_bytes':len(result.stdout),'weights_or_activation_archives_transferred':False},indent=2))
    assert process.get('exit_code')==0 and current.get('completed'), 'Failed/unfinished execution retained; do not relaunch blindly'


if __name__=='__main__':
    try:main()
    except Exception as exc:print(str(exc),file=sys.stderr);sys.exit(1)
