"""Selective private bootstrap for the R126 full-generation regression only."""
from __future__ import annotations
import multiprocessing as mp
import multiprocessing.process as process_module
import os
from pathlib import Path

_INSTALLED=False
_BOOTSTRAP_PATCHED=False


def install_alignment():
    global _INSTALLED
    if _INSTALLED:return
    previous=os.environ.pop('H3_R126_FULL_SELECTIVE_BOOTSTRAP',None)
    try:
        from integration_hooks126 import install
        install()
        _INSTALLED=True
    finally:
        if previous is not None:os.environ['H3_R126_FULL_SELECTIVE_BOOTSTRAP']=previous


def configure_python_startup():
    global _BOOTSTRAP_PATCHED
    key_path=Path(os.environ['H3_R126_IPC_KEY_FILE']).resolve()
    assert key_path.parent==Path('/tmp/h3_staging143') and key_path.stat().st_mode&0o077==0
    key=key_path.read_bytes();assert len(key)==32
    mp.current_process().authkey=key
    if _BOOTSTRAP_PATCHED:return
    original=process_module.BaseProcess._bootstrap
    def bootstrap(self,*args,**kwargs):
        if self.name.startswith('DiffusionWorker-'):install_alignment()
        return original(self,*args,**kwargs)
    process_module.BaseProcess._bootstrap=bootstrap
    _BOOTSTRAP_PATCHED=True
