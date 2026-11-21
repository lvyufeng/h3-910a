"""Private R127 authenticated multiprocessing; never import model helpers unconditionally."""
import os
if os.environ.get('H3_R127_IPC_KEY_FILE'):
    from pathlib import Path
    import multiprocessing as mp
    path=Path(os.environ['H3_R127_IPC_KEY_FILE']).resolve()
    assert path.parent==Path('/tmp/h3_runtime12701') and path.stat().st_mode&0o077==0
    key=path.read_bytes();assert len(key)==32;mp.current_process().authkey=key
if os.environ.get('H3_R127_SELECTIVE_BOOTSTRAP')=='1':
    from full_bootstrap127 import configure_python_startup
    configure_python_startup()
