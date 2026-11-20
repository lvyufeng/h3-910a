"""Isolated worker bootstrap; active only for this experiment's NPU generation.

Imported by worker Python startup through an explicit private PYTHONPATH. It grants
no permissions and leaves shared files untouched. All precision routes and use of
exported initial noise are recorded. The NPU encoder itself always runs live.
"""
import os
if os.environ.get('H3_NPU_ALIGNMENT124_SELECTIVE_BOOTSTRAP')=='1':
    from npu_process_bootstrap124e import configure_python_startup
    configure_python_startup()
elif os.environ.get('H3_NPU_ALIGNMENT124_BOOTSTRAP')=='1':
    # vLLM imports probe backends through short-lived Python subprocesses.
    # Do not recursively install the full pipeline into those import probes.
    # Restore the opt-in only after installation so actual engine workers
    # still install their own worker-local integration.
    os.environ.pop('H3_NPU_ALIGNMENT124_BOOTSTRAP')
    try:
        from alignment_hooks124 import install
        install()
    finally:
        os.environ['H3_NPU_ALIGNMENT124_BOOTSTRAP']='1'
