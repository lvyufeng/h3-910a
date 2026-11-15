#!/usr/bin/env python3
"""A/B core-count scheduling, identical FA math and tiles, private fresh builds.

Control 32, candidate 30 (DCMI physical core count), and 16 test the second-wave
hypothesis. Production source/cache are never overwritten. Every variant compiles
under a unique name in a new private directory. Numerical comparison uses the same
inputs and sampled full-sequence FP32 softmax attention as an independent reference.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from h3_core_audit_probe import DCMI, load_backend, measure


def build(cores, name, work):
    import tbe.common.platform as plat
    plat.set_current_compile_soc_info("Ascend910A")
    sys.path.insert(0, "/tmp/fa_pkg")
    from flash_attention.ops.flash_attention.flash_attention_fwd import FlashAttentionFwd
    from flash_attention.ops.flash_attention.tiling_strategy.strategy import TilingStrategy
    b, h, n, d = 1, 7, 63232, 128
    nz = [b, h, d // 16, n // 16, 16, 16]
    def desc(ori, dt, fmt="NZ"):
        return {"shape": nz[:], "ori_shape": ori[:], "dtype": dt, "format": fmt}
    old = os.getcwd()
    os.chdir(work)
    try:
        os.environ["FA_BR"] = "128"
        os.environ["FA_BC"] = "304"
        fa = FlashAttentionFwd(q=desc([b, h, n, d], "float16"), k=desc([b, h, n, d], "float16"),
                              v=desc([b, h, n, d], "float16"), attn_mask=None, dropout_mask=None,
                              alibi_mask=None, kernel_name=name,
                              tiling_stgy=TilingStrategy.from_strategy_name("sparse"), disable_debug=True)
        fa.core_num = cores
        print(f"Fresh compile: core_num={cores}, name={name}, directory={work}", flush=True)
        fa.compute_process()
        obj = Path(work) / "kernel_meta" / (name + ".o")
        meta = json.loads(obj.with_suffix(".json").read_text())
        assert meta["blockDim"] == cores, meta
        return {"object": str(obj), "kernel": meta["kernelName"], "block_dim": cores,
                "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(), "metadata": meta}
    finally:
        os.chdir(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--dwell", type=float, default=2)
    args = ap.parse_args()
    p = Path(args.output)
    if p.exists():
        raise RuntimeError("Refuse to overwrite prior audit evidence")
    dcmi = DCMI(0)
    static = dcmi.static()
    if static["physical_aicore_count"] != 30:
        raise RuntimeError("A/B designed for this verified 30-core device; re-evaluate on a different device")
    work = tempfile.mkdtemp(prefix="h3_corecount_ab_")
    result = {"static": static, "build_dir": work, "kernels": {}, "runs": [], "validation": {}}
    source_files = ["attention.py", "flash_attention_fwd.py", "tik_ops_utils.py"]
    result["source_sha256"] = {f: hashlib.sha256((Path("/tmp/fa_pkg/flash_attention/ops/flash_attention") / f).read_bytes()).hexdigest()
                                for f in source_files}
    for cores in (32, 30, 16):
        result["kernels"][str(cores)] = build(cores, f"fa_coreaudit_c{cores}_br128_bc304", work)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch.manual_seed(91)
    b, h, n, d = 1, 7, 63232, 128
    shape = (b, h, d // 16, n // 16, 16, 16)
    q, k, v = [torch.randn(shape, dtype=torch.float16, device="npu") * 0.05 for _ in range(3)]
    l = torch.empty((b, h, n), dtype=torch.float32, device="npu")
    outputs = {cores: torch.empty_like(q) for cores in (32, 30, 16)}
    fa = load_backend()
    lib = fa._lib()
    st = C.c_void_p(torch.npu.current_stream().npu_stream)
    funcs = {}
    for cores in (32, 30, 16):
        entry = result["kernels"][str(cores)]
        handle = lib.fa910a_register(entry["object"].encode(), entry["kernel"].encode())
        if not handle:
            raise RuntimeError(f"Registration failed for {cores}")
        def launch(cores=cores, entry=entry, handle=handle):
            rc = lib.fa910a_launch(handle, entry["kernel"].encode(), cores,
                                  q.data_ptr(), k.data_ptr(), v.data_ptr(), outputs[cores].data_ptr(), l.data_ptr(), st)
            if rc:
                raise RuntimeError(f"Launch failed for {cores}, rc={rc}")
        funcs[cores] = launch
    # Two rounds, reversed order controls for thermal/order bias.
    for round_id, order in enumerate(((32, 30, 16), (16, 30, 32))):
        for cores in order:
            run = measure(torch, dcmi, f"fresh FA cores={cores}, round={round_id}", funcs[cores],
                          4 * h * n * n * d, args.dwell, iterations=2)
            run["cores"] = cores
            result["runs"].append(run)
    torch.npu.synchronize()
    for cores in (30, 16):
        delta = (outputs[cores].float() - outputs[32].float()).abs()
        result["validation"][f"c{cores}_vs_c32"] = {
            "max_abs_diff": delta.max().item(), "all_finite": bool(torch.isfinite(outputs[cores]).all().item()),
            "bit_exact": bool(torch.equal(outputs[cores], outputs[32]))}
    def nd(x):
        return x.permute(0, 1, 3, 4, 2, 5).reshape(b, h, n, d).contiguous()
    sample = torch.tensor([0, 127, 128, 14719, 31615, 60000, 63104, 63231], dtype=torch.long, device="npu")
    q_nd, k_nd, v_nd = nd(q), nd(k), nd(v)
    # Raw kernel deliberately has no scale: reference has scale=1.
    ref = torch.softmax(q_nd[:, :, sample].float() @ k_nd.float().transpose(-1, -2), dim=-1) @ v_nd.float()
    for cores in (32, 30, 16):
        actual = nd(outputs[cores])[:, :, sample].float()
        diff = actual - ref
        result["validation"][f"c{cores}_vs_fp32_sample"] = {
            "sample_rows": sample.cpu().tolist(), "max_abs_error": diff.abs().max().item(),
            "rms_relative_error": (diff.square().mean().sqrt() / ref.square().mean().sqrt()).item()}
    p.write_text(json.dumps(result, indent=2) + "\n")
    print("Validation:", json.dumps(result["validation"]), flush=True)
    print("Saved fresh-build A/B evidence:", p, flush=True)


if __name__ == "__main__":
    main()
