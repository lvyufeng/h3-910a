#!/usr/bin/env python3
"""Dry-run ``build_kernel`` against a FAKE TIK/CANN, to prove spec assembly only.

This is NOT a compile. It replaces ``tbe.tik`` and ``tbe.common.platform`` with
in-memory fakes so the real ``build_kernel`` code path (plan -> emit -> BuildCCE ->
kernel_meta copy -> spec103.json -> cache hit -> block_dim guard) executes with no
CANN, no compiler, no NPU and no torch. The written .o is a placeholder blob; the
SHA256 is therefore a *file* hash, not a binary identity.

What this proves: the metadata dict contains every key the root engine and
Runtime5GM read (obj_path, obj_sha256, kernel_symbol, block_dim), that the object
SHA check is self-consistent, that block_dim equals the launched core count, that
`spec103.json` is valid JSON, and that a second call is served from cache.
What this does NOT prove: that the emitted program compiles or runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_flash():
    spec = importlib.util.spec_from_file_location("flash_vision103_dryrun",
                                                  HERE / "flash_vision103.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sym:
    """A symbolic loop counter; it only has to survive the emitter's arithmetic."""

    __slots__ = ("expr",)

    def __init__(self, expr):
        self.expr = expr

    def __add__(self, other):
        return Sym("(%s+%r)" % (self.expr, other))

    __radd__ = __add__

    def __sub__(self, other):
        return Sym("(%s-%r)" % (self.expr, other))

    def __mul__(self, other):
        return Sym("(%s*%r)" % (self.expr, other))

    __rmul__ = __mul__

    def __floordiv__(self, other):
        return Sym("(%s//%r)" % (self.expr, other))

    def __eq__(self, other):
        return Sym("(%s==%r)" % (self.expr, other))

    def __hash__(self):
        return object.__hash__(self)

    def __lt__(self, other):
        return Sym("(%s<%r)" % (self.expr, other))

    def __repr__(self):
        return "Sym(%s)" % self.expr


class _Ctx:
    def __init__(self, tik, var=False):
        self.tik = tik
        self.var = var

    def __enter__(self):
        return Sym("loop_v") if self.var else None

    def __exit__(self, *_exc):
        return False


class _Tensor:
    def __init__(self, tick, dtype, shape, name="", scope=""):
        self.tik = tick
        self.dtype = dtype
        self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        self.name = name
        self.scope = scope
        tick.tensors.append(self)

    def reshape(self, shape):
        return _Tensor(self.tik, self.dtype, shape, name=self.name + ".rs",
                       scope=self.scope)

    def reinterpret_cast_to(self, dtype):
        return _Tensor(self.tik, dtype, self.shape, name=self.name + ".re",
                       scope=self.scope)

    def __getitem__(self, index):
        return _Slice(self, index)


class _Slice:
    def __init__(self, tensor, index):
        self.tensor = tensor
        self.index = index

    def __getitem__(self, index):
        return self


class _Scalar:
    def __init__(self, *a, **kw):
        pass

    def set_as(self, value):
        pass


class FakeTik:
    """Records nothing; just satisfies every call the emitter makes."""

    scope_gm = "gm"
    scope_cbuf = "cbuf"
    scope_cc = "cc"
    scope_ubuf = "ubuf"

    def __init__(self, disable_debug=True):
        self.tensors = []
        self.build = None
        self.ops = 0
        self.block_dim = None

    def Tensor(self, dtype, shape, name="", scope=""):
        return _Tensor(self, dtype, shape, name=name, scope=scope)

    def Scalar(self, *a, **kw):
        return _Scalar()

    def for_range(self, begint, endt, block_num=None):
        if block_num:
            self.block_dim = block_num
        return _Ctx(self, var=True)

    def if_scope(self, cond):
        return _Ctx(self)

    def new_stmt_scope(self, disable_sync=False):
        return _Ctx(self)

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)

        def _call(*args, **kwargs):
            self.ops += 1

        return _call

    def BuildCCE(self, kernel_name, inputs, outputs):
        self.build = {"kernel_name": kernel_name,
                      "inputs": [t.name for t in inputs],
                      "outputs": [t.name for t in outputs]}
        meta_dir = Path.cwd() / "kernel_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / (kernel_name + ".o")).write_bytes(
            b"FAKE-OBJECT-NOT-A-BINARY " + kernel_name.encode())
        (meta_dir / (kernel_name + ".json")).write_text(json.dumps(
            {"kernelName": kernel_name, "blockDim": self.block_dim,
             "coreType": "AiCore"}))


def install_fakes():
    tbe = types.ModuleType("tbe")
    tik = types.ModuleType("tbe.tik")
    tik.Tik = FakeTik
    tik.scope_gm = FakeTik.scope_gm
    tik.scope_cbuf = FakeTik.scope_cbuf
    tik.scope_cc = FakeTik.scope_cc
    tik.scope_ubuf = FakeTik.scope_ubuf
    common = types.ModuleType("tbe.common")
    platform = types.ModuleType("tbe.common.platform")

    def set_soc(_name):
        return None

    def get_soc_spec(field):
        return {"SHORT_SOC_VERSION": "Ascend910", "CORE_NUM": 32,
                "UB_SIZE": 262144, "L1_SIZE": 1048576,
                "L0C_SIZE": 262144}.get(field)

    def intrinsic_check_support(name, dtype):
        return (name, dtype) not in (("Intrinsic_vcmax", "float32"),
                                     ("Intrinsic_vexp", "float16"),
                                     ("Intrinsic_vtranspose", "float32"),
                                     ("Intrinsic_vtranspose", "float32"),
                                     ("Intrinsic_vbrcb", "uint16"))

    platform.set_current_compile_soc_info = set_soc
    platform.get_soc_spec = get_soc_spec
    platform.intrinsic_check_support = intrinsic_check_support
    tbe.tik = tik
    tbe.common = common
    common.platform = platform
    sys.modules["tbe"] = tbe
    sys.modules["tbe.tik"] = tik
    sys.modules["tbe.common"] = common
    sys.modules["tbe.common.platform"] = platform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    flash = load_flash()
    install_fakes()
    report = {"scope": __doc__, "fake_sdk": True, "cann_used": False,
              "compiled": False, "executed": False, "not_an_accuracy_or_speed_result": True}
    with tempfile.TemporaryDirectory(prefix="dryrun103_") as tmp:
        out_dir = Path(tmp) / "kernels"
        meta = flash.build_kernel(batch=16, m=4032, n=4032, d=72, out_dir=out_dir)
        spec_path = Path(meta["obj_path"]).parent / "spec103.json"
        spec = json.loads(spec_path.read_text())
        required = ["obj_path", "obj_sha256", "kernel_symbol", "block_dim",
                    "GM_argument_count", "kernel_arg_order", "logical", "grid",
                    "dma", "buffers", "vector", "math", "host_protocol",
                    "assumptions", "capability", "source_sha256"]
        report["real_geometry"] = {
            "kernel_name": meta["kernel_name"],
            "kernel_symbol": meta["kernel_symbol"],
            "block_dim": meta["block_dim"],
            "GM_argument_count": meta["GM_argument_count"],
            "obj_sha256_self_consistent": sha(meta["obj_path"]) == meta["obj_sha256"],
            "obj_sha256": meta["obj_sha256"],
            "source_sha256": meta["source_sha256"],
            "all_required_metadata_keys_present":
                all(key in meta for key in required),
            "missing_keys": [key for key in required if key not in meta],
            "runtime5gm_keys_present": all(
                key in meta for key in ("block_dim", "kernel_symbol", "obj_path",
                                        "obj_sha256")),
            "engine_block_dim_assert_30": meta["block_dim"] == 30,
            "spec103_executed_flag": spec["executed"],
            "spec103_not_a_speed_claim": spec["not_a_speed_claim"],
            "spec103_numerical_tolerance": spec["numerical_tolerance"],
            "spec103_round": spec["round"],
            "grid": meta["grid"],
            "ub_total_bytes": meta["buffers"]["ub_total_bytes"],
            "l0c_total_bytes": meta["buffers"]["l0c_total_bytes"],
            "l1_total_bytes": meta["buffers"]["l1_total_bytes"],
            "capability_compile_soc": meta["capability"]["compile_soc"],
            "capability_all_required_supported":
                meta["capability"]["all_required_supported"],
        }
        cached = flash.build_kernel(batch=16, m=4032, n=4032, d=72, out_dir=out_dir)
        report["cache_hit"] = {
            "same_obj_path": cached["obj_path"] == meta["obj_path"],
            "same_obj_sha256": cached["obj_sha256"] == meta["obj_sha256"],
            "compiled_now": cached["compiled_now"],
        }
        smoke = flash.build_kernel(batch=2, m=64, n=192, d=72,
                                   out_dir=Path(tmp) / "smoke_kernels")
        report["smoke_geometry"] = {
            "kernel_name": smoke["kernel_name"],
            "block_dim": smoke["block_dim"],
            "grid": smoke["grid"],
            "logical_n_pad": smoke["logical"]["n_pad"],
            "phantom_keys": smoke["logical"]["phantom_keys"],
            "obj_sha256_self_consistent":
                sha(smoke["obj_path"]) == smoke["obj_sha256"],
        }
        # Tamper the cached metadata: the block_dim guard must refuse a reuse whose
        # blockDim no longer matches the plan (this is the guard the engine relies on).
        smoke_meta_path = Path(smoke["obj_path"]).with_suffix(".json")
        tampered = json.loads(smoke_meta_path.read_text())
        tampered["blockDim"] = 30
        smoke_meta_path.write_text(json.dumps(tampered))
        guard = {}
        try:
            flash.build_kernel(batch=2, m=64, n=192, d=72,
                               out_dir=Path(tmp) / "smoke_kernels")
            guard["refused"] = False
        except flash.FlashVisionError as exc:
            guard["refused"] = True
            guard["message"] = str(exc)
        report["block_dim_guard"] = guard
        # and the plan-level specialisation guards
        plan_guards = {}
        for kwargs in ({"mb": 64}, {"bc": 128}, {"m": 48}, {"d": 64}):
            try:
                flash.plan_kernel(**kwargs)
                plan_guards[str(kwargs)] = False
            except flash.FlashVisionError:
                plan_guards[str(kwargs)] = True
        report["plan_refusals"] = plan_guards
    out = HERE / "dryrun103.json"
    if out.is_file():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["real_geometry"], indent=2, sort_keys=True))
    print("cache_hit", report["cache_hit"])
    print("smoke", report["smoke_geometry"])
    print("block_dim_guard", report["block_dim_guard"])
    print("plan_refusals", report["plan_refusals"])


if __name__ == "__main__":
    main()
