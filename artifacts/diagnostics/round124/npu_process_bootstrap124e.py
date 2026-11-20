"""Experiment-local Python IPC/bootstrap integration for the original NPU H3 run.

CANN unpickles authenticated multiprocessing manager proxies in Python helper
processes. Give only this experiment's descendants the same random private
session key; do not disable authentication, print the key, connect to other
managers, or alter shared SDK files. Full model hooks run only in explicit driver
calls and named vLLM DiffusionWorker processes, never CANN/compiler/import probes.
"""
from __future__ import annotations
import multiprocessing as mp
import multiprocessing.process as process_module
import os
from pathlib import Path

_INSTALLED=False
_PROCESS_HOOK_INSTALLED=False


def install_alignment():
    global _INSTALLED
    if _INSTALLED:return
    # Prevent backend discovery Python children from loading H3 model hooks.
    previous=os.environ.pop('H3_NPU_ALIGNMENT124_BOOTSTRAP',None)
    try:
        from alignment_hooks124 import install
        install()
        _INSTALLED=True
    finally:
        if previous is not None:os.environ['H3_NPU_ALIGNMENT124_BOOTSTRAP']=previous


def configure_python_startup():
    global _PROCESS_HOOK_INSTALLED
    key_path=Path(os.environ['H3_NPU_ALIGNMENT124_IPC_KEY_FILE'])
    assert key_path.resolve().parent==Path('/tmp/h3_align12401')
    assert key_path.stat().st_mode & 0o077 == 0
    key=key_path.read_bytes()
    assert len(key)==32
    mp.current_process().authkey=key
    if _PROCESS_HOOK_INSTALLED:return
    original=process_module.BaseProcess._bootstrap
    def bootstrap(self,*args,**kwargs):
        if self.name.startswith('DiffusionWorker-'):
            install_alignment()
        return original(self,*args,**kwargs)
    process_module.BaseProcess._bootstrap=bootstrap
    _PROCESS_HOOK_INSTALLED=True
