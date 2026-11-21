#!/usr/bin/env python3
"""Collect bounded R129 diagnostic JSON/text only, never model or activation files."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

SCRIPT=r'''
from pathlib import Path
import io,json,os,select,sys,tarfile
root=Path('/tmp/h3_repeat12901');session=root/'engine_repeat129a';receipt=root/'engine_repeat129a_process.json'
r=json.loads(receipt.read_text());assert r['pid']==int(sys.argv[1])
if r.get('exit_code') is None:
 proc=Path('/proc',str(r['launcher_pid']))
 if proc.exists():
  assert b'launch_diagnostics129.py' in (proc/'cmdline').read_bytes()
  fd=os.pidfd_open(r['launcher_pid'])
  try:
   ready,_,_=select.select([fd],[],[],7500);assert ready
  finally:os.close(fd)
 r=json.loads(receipt.read_text());assert r.get('exit_code') is not None
files=[receipt,session/'diagnostic_lifecycle129.json',root/'launch_source_pins129a.json']
files+=list(root.glob('*_rank*129.json'))+list(root.glob('cohort_failure*.json'))
files+=list(session.glob('*.jsonl'))+list(session.glob('initialization_crash_pid*127.txt'))
assert all(p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.txt') for p in files)
assert sum(p.stat().st_size for p in files)<=32*1024**2
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=6) as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    assert not args.output_dir.exists()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=20',
        '-o','ControlMaster=no','-o','ControlPath=none','ts-217',f'/usr/bin/python3 - {args.pid}']
    result=subprocess.run(command,input=SCRIPT.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    assert len(result.stdout)<=32*1024**2;args.output_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as tar:
        members=tar.getmembers();assert sum(m.size for m in members)<=32*1024**2
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).suffix in ('.json','.jsonl','.txt') for m in members)
        tar.extractall(args.output_dir,filter='data')
    process=json.loads((args.output_dir/'engine_repeat129a_process.json').read_text());lifecycle=json.loads((args.output_dir/'engine_repeat129a/diagnostic_lifecycle129.json').read_text())
    print(json.dumps({'exit_code':process['exit_code'],'completed':lifecycle['completed'],'diagnostic_RPCs':lifecycle['RPCs'],
        'failure':lifecycle.get('failure'),'compressed_evidence_bytes':len(result.stdout),'model_activation_files_transferred':False},indent=2))


if __name__=='__main__':main()
