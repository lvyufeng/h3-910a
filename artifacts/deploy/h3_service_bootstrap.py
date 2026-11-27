"""Process-startup bootstrap for the MiniMax-H3 service.

It reproduces the verified R124/R126 startup contract:

* a private multiprocessing authkey, so every worker of the engine shares exactly
  one IPC credential;
* when ``H3_R126_FULL_SELECTIVE_BOOTSTRAP=1``, the same ``BaseProcess._bootstrap``
  hook that makes every ``DiffusionWorker-*`` child call
  ``full_bootstrap126.install_alignment()`` -- this is what keeps the numeric
  path identical to the reference runs;

and then appends the deployment speedups from :mod:`h3_optimizations` to that
same hook, so they are installed *inside each worker process* as well.
"""
from __future__ import annotations

import os
from pathlib import Path

_CONFIGURED = False


def configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    key_file = os.environ.get("H3_R126_IPC_KEY_FILE")
    if key_file:
        import multiprocessing as mp

        key_path = Path(key_file).resolve()
        key = key_path.read_bytes()
        if len(key) != 32:
            raise RuntimeError(f"IPC authkey must be 32 bytes, got {len(key)}")
        mp.current_process().authkey = key

    if os.environ.get("H3_R126_FULL_SELECTIVE_BOOTSTRAP") == "1":
        from full_bootstrap126 import configure_python_startup

        configure_python_startup()
        _wrap_integration_hooks()


def _wrap_integration_hooks() -> None:
    """Append the service speedups to whatever the alignment hook installs."""
    import integration_hooks126

    if getattr(integration_hooks126.install, "_h3_service_wrapped", False):
        return
    original_install = integration_hooks126.install

    def install():
        original_install()
        try:
            from h3_optimizations import install as install_optimizations

            install_optimizations(verbose=True)
        except Exception as exc:  # never break the numeric alignment path
            print(f"[h3-boot] speedups skipped: {exc!r}", flush=True)
        _harden_repeat_requests()

    install._h3_service_wrapped = True
    integration_hooks126.install = install


def _harden_repeat_requests() -> None:
    """Make the reference capture hook safe for many requests per process.

    ``integration_hooks126`` writes ``live_NPU_encoder_hidden_pid<pid>126.pt`` with
    ``Path.open("xb")`` -- one capture per process, which is true for the one-shot
    reference run but raises ``FileExistsError`` on the second service request.
    Rotating the capture root per encoder call keeps the artifacts (they are
    experiment-only) without ever colliding.
    """
    try:
        import itertools

        import integration_hooks126 as hooks
        import vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 as pipeline
    except Exception:
        return
    encoder_cls = getattr(pipeline, "MiniMaxH3Qwen3VLEncoder", None)
    if encoder_cls is None:
        return
    inner = encoder_cls._encode
    if getattr(inner, "_h3_service_rotated", False):
        return
    counter = itertools.count()
    base = Path(hooks.GENERATION_ROOT)

    def _encode_rotated(self, *args, **kwargs):
        target = base / "requests" / f"pid{os.getpid()}_{next(counter)}"
        target.mkdir(parents=True, exist_ok=True)
        hooks.GENERATION_ROOT = target
        return inner(self, *args, **kwargs)

    _encode_rotated._h3_service_rotated = True
    encoder_cls._encode = _encode_rotated


def install_parent_alignment() -> None:
    """Install alignment in the service (parent) process, like the reference entry."""
    from full_bootstrap126 import install_alignment

    install_alignment()


def install_plain_speedups() -> None:
    """Non-omni entry points: just install the speedups in this process."""
    try:
        from h3_optimizations import install as install_optimizations

        install_optimizations(verbose=True)
    except Exception as exc:
        print(f"[h3-boot] speedups skipped: {exc!r}", flush=True)