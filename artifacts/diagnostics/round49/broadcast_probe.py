#!/usr/bin/env python3
"""Compile/run exact FP32 row-broadcast multiplication using TIK's high-level API.

No production changes. Check whether h_mul(matrix, matrix, column) lowers the
broadcast correctly, before changing the full FA kernel.
"""
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from tbe import tik
import tbe.common.platform as plat
plat.set_current_compile_soc_info("Ascend910A")
M, N = 128, 16
work = Path(tempfile.mkdtemp(prefix="h3_rowbroadcast_probe_"))
import os
os.chdir(work)
result = {"directory": str(work), "M": M, "N": N}
for variant in ("scalar", "column"):
    try:
        t = tik.Tik(disable_debug=True)
        q = t.Tensor("float32", (M, N), name="q", scope=tik.scope_gm)
        k = t.Tensor("float32", (M,), name="k", scope=tik.scope_gm)
        v = t.Tensor("float32", (M, N), name="v_unused", scope=tik.scope_gm)
        o = t.Tensor("float32", (M, N), name="o", scope=tik.scope_gm)
        l = t.Tensor("float32", (M,), name="l_unused", scope=tik.scope_gm)
        matrix = t.Tensor("float32", (M, N), name="matrix", scope=tik.scope_ubuf)
        col = t.Tensor("float32", (M,), name="column", scope=tik.scope_ubuf)
        t.data_move(matrix, q, 0, 1, M * N // 8, 0, 0)
        t.data_move(col, k, 0, 1, M // 8, 0, 0)
        if variant == "scalar":
            broad = t.Tensor("float32", (M, N), name="broad", scope=tik.scope_ubuf)
            s = t.Scalar("float32", name="coefficient")
            with t.for_range(0, M) as row:
                s.set_as(col[row])
                t.h_duplicate(broad[row, :], s)
            t.h_mul(matrix, matrix, broad)
        else:
            t.h_mul(matrix, matrix, col.reshape((M, 1)))
        t.data_move(o, matrix, 0, 1, M * N // 8, 0, 0)
        name = f"rowbroadcast_{variant}"
        t.BuildCCE(kernel_name=name, inputs=[q, k, v], outputs=[o, l],
                   config={"dump_cce_code": False, "save_temp_cce_file": True})
        result[variant] = {"built": True, "object": str(work / "kernel_meta" / f"{name}.o")}
    except Exception as exc:
        result[variant] = {"built": False, "error": str(exc)}
        print(f"Build failed for {variant}: {exc}", flush=True)

import torch
import torch_npu
import time
torch.npu.set_device(0)
fa_path = "/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py"
s = importlib.util.spec_from_file_location("fa_broadcast_audit", fa_path)
fa = importlib.util.module_from_spec(s)
s.loader.exec_module(fa)
lib = fa._lib()
q = torch.randn(M, N, dtype=torch.float32, device="npu")
k = torch.rand(M, dtype=torch.float32, device="npu")
v = torch.empty_like(q)
l = torch.empty(M, dtype=torch.float32, device="npu")
reference = q * k[:, None]
outputs = {}
for variant in ("scalar", "column"):
    if not result[variant]["built"]:
        continue
    entry = result[variant]
    o = torch.empty_like(q)
    meta = json.loads(Path(entry["object"]).with_suffix(".json").read_text())
    name = meta["kernelName"]
    handle = lib.fa910a_register(entry["object"].encode(), name.encode())
    assert handle
    stream = C.c_void_p(torch.npu.current_stream().npu_stream)
    def call():
        rc = lib.fa910a_launch(handle, name.encode(), meta["blockDim"],
                              q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), l.data_ptr(), stream)
        if rc:
            raise RuntimeError(f"launch failed rc={rc}")
    for _ in range(3):
        call()
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        call()
    torch.npu.synchronize()
    entry["microseconds"] = (time.perf_counter() - start) * 1e6 / 100
    entry["max_abs_error"] = (o - reference).abs().max().item()
    entry["bit_exact_to_reference"] = bool(torch.equal(o, reference))
    outputs[variant] = o
if len(outputs) == 2:
    result["same_bits"] = bool(torch.equal(outputs["scalar"], outputs["column"]))
(work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
