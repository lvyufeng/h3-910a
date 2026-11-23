"""Authenticated IPC only for the private R126 Python path; never load models here."""
import os
if os.environ.get('H3_R126_IPC_KEY_FILE'):
    import multiprocessing as mp
    from pathlib import Path
    key_path = Path(os.environ['H3_R126_IPC_KEY_FILE']).resolve()
    assert key_path.parent == Path('/tmp/h3_budget149')
    assert key_path.stat().st_mode & 0o077 == 0
    key = key_path.read_bytes()
    assert len(key) == 32
    mp.current_process().authkey = key
if os.environ.get('H3_R126_FULL_SELECTIVE_BOOTSTRAP')=='1':
    from full_bootstrap126 import configure_python_startup
    configure_python_startup()
