"""Deployment sitecustomize: loaded in every Python process of the service.

Placed first on PYTHONPATH so it shadows the reference checkout's own
sitecustomize; it performs the same startup steps and then adds the speedups.
Set ``H3_SERVICE_SKIP_BOOTSTRAP=1`` to load nothing at all (debugging).
"""
import os

if os.environ.get("H3_SERVICE_SKIP_BOOTSTRAP") != "1":
    try:
        import h3_service_bootstrap

        h3_service_bootstrap.configure()
        if os.environ.get("H3_R126_FULL_SELECTIVE_BOOTSTRAP") != "1":
            h3_service_bootstrap.install_plain_speedups()
    except Exception as _exc:  # pragma: no cover - startup must never hard-fail here
        print(f"[h3-sitecustomize] bootstrap skipped: {_exc!r}", flush=True)