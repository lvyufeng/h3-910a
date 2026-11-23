"""Authenticate and install only in real DiffusionWorker processes, not compiler helpers."""
from __future__ import annotations
import multiprocessing as mp
import multiprocessing.process as processes
import os
from pathlib import Path
_INSTALLED=False
_PATCHED=False


def install_alignment():
    global _INSTALLED
    if _INSTALLED:return
    previous=os.environ.pop('H3_R127_SELECTIVE_BOOTSTRAP',None)
    try:
        from integration_hooks127 import install
        install();_INSTALLED=True
    finally:
        if previous is not None:os.environ['H3_R127_SELECTIVE_BOOTSTRAP']=previous


def configure_python_startup():
    global _PATCHED
    key_path=Path(os.environ['H3_R127_IPC_KEY_FILE']).resolve()
    assert key_path.parent==Path('/tmp/h3_vae13901') and not key_path.is_symlink()
    assert key_path.stat().st_mode&0o077==0
    key=key_path.read_bytes();assert len(key)==32
    mp.current_process().authkey=key
    if _PATCHED:return
    original=processes.BaseProcess._bootstrap
    def bootstrap(self,*args,**kwargs):
        if self.name.startswith('DiffusionWorker-'):install_alignment()
        return original(self,*args,**kwargs)
    processes.BaseProcess._bootstrap=bootstrap;_PATCHED=True
