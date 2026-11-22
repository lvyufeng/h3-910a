#!/usr/bin/env python3
"""Collect bounded scalar component receipts; never encoder inputs or raw outputs."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

SCRIPT=r'''
from pathlib import Path
import io,json,os,select,sys,tarfile
root=Path('/tmp/h3_components13701');session=root/'engine_components137a';receipt=root/'engine_components137a_process.json'
r=json.loads(receipt.read_text());assert r['pid']==int(sys.argv[1])
if r.get('exit_code') is None and not r.get('completion_unknown'):
 proc=Path('/proc',str(r['launcher_pid']))
 if proc.exists():
  assert b'launch_components137.py' in (proc/'cmdline').read_bytes()
  fd=os.pidfd_open(r['launcher_pid'])
  try:ready,_,_=select.select([fd],[],[],7500);assert ready
  finally:os.close(fd)
 r=json.loads(receipt.read_text());assert r.get('exit_code') is not None or r.get('completion_unknown')
files=[receipt,session/'component_lifecycle137.json',root/'launch_source_pins137a.json']
files+=list(root.glob('component_rank*137.json'))+list(root.glob('cohort_failure*.json'))+list(root.glob('private_abort_details137*.json'))
files+=list(session.glob('*.jsonl'))+list(session.glob('initialization_crash_pid*127.txt'))
assert all(p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.txt') for p in files)
assert sum(p.stat().st_size for p in files)<=64*1024**2
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=6) as archive:
 for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    assert not args.output_dir.exists()
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=60','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=20',
        '-o','ControlMaster=no','-o','ControlPath=none','ts-217',f'/usr/bin/python3 - {args.pid}'],input=SCRIPT.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    assert len(result.stdout)<=64*1024**2;args.output_dir.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        members=archive.getmembers();assert sum(m.size for m in members)<=64*1024**2
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and Path(m.name).suffix in ('.json','.jsonl','.txt') for m in members)
        archive.extractall(args.output_dir,filter='data')
    process=json.loads((args.output_dir/'engine_components137a_process.json').read_text());life=json.loads((args.output_dir/'engine_components137a/component_lifecycle137.json').read_text())
    print(json.dumps({'exit_code':process.get('exit_code'),'completed':life['completed'],'failure':life.get('failure'),
        'compressed_scalar_evidence_bytes':len(result.stdout),'model_weights_or_input_activation_archives_transferred':False},indent=2))


if __name__=='__main__':main()
