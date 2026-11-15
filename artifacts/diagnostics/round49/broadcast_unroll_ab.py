#!/usr/bin/env python3
"""Controlled FA experiment: unroll only the per-row FP32 scale broadcast.

The hardware trace reports ~22.4% scalar and 55.1% vector pipe ratios. TIK h_mul
rejects last-axis column broadcasting, so test removing runtime loop-index/branch
work from the existing exact scalar broadcast, without changing precision/math.
Only a private new binary is built; production source/cache are left unchanged.
"""
from __future__ import annotations
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
from tbe import tik
import tbe.common.platform as plat

plat.set_current_compile_soc_info("Ascend910A")
sys.path.insert(0, "/tmp/fa_pkg")
from flash_attention.ops.flash_attention.flash_attention_fwd import FlashAttentionFwd
from flash_attention.ops.flash_attention.constants import FP32, UB
from flash_attention.ops.flash_attention.tiling_strategy.strategy import TilingStrategy


def output_compute_unrolled(self, Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, kv_blk_h):
    m = Pij_l1_zN.shape[1]
    with self.tik_instance.new_stmt_scope(disable_sync=False):
        Oij_ub_zN = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, q_blk_h, kv_blk_h, self.actual_d,
                                                  N1MN0_to_MN=False, precision_type=FP32)
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.cont_data_mv_1_bust(dst=Oi_l1_zN, src=Oij_ub_zN, burst=m * self.d // 8)
        with self.tik_instance.else_scope():
            broadcast_scale_ub = self.tik_instance.Tensor(FP32, (1, m, self.N0), scope=UB, name="broadcast_scale_ub")
            src_scalar = self.tik_instance.Scalar(FP32, name="src_scalar")
            # Only change versus the inspected source: Python/static unroll instead
            # of TIK for_range(0,m). Scalar values and broadcast are identical.
            for idx in range(m):
                src_scalar.set_as(update_scale_ub[idx])
                self.tik_instance.h_duplicate(broadcast_scale_ub[0, idx, :], src_scalar)
            single_loop_size = 8
            loop_times, tail_num = divmod(self.N1, single_loop_size)
            with self.tik_instance.for_range(0, loop_times) as idx:
                with self.tik_instance.for_range(0, 2, thread_num=2) as t_idx:
                    single_thread_size = single_loop_size // 2
                    offset = idx * single_loop_size * m * self.N0 + t_idx * single_thread_size * m * self.N0
                    n1_begin = idx * single_loop_size + t_idx * single_thread_size
                    self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub,
                                       offset, n1_begin, single_thread_size, m)
            if tail_num > 0:
                offset = loop_times * single_loop_size * m * self.N0
                thread_num = min(tail_num, 2)
                n1_begin = loop_times * single_loop_size
                n1_size1 = tail_num // thread_num
                self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub, offset, n1_begin, n1_size1, m)
                n1_size2 = tail_num - n1_size1
                if n1_size2 > 0:
                    offset = offset + (tail_num // thread_num) * m * self.N0
                    n1_begin = loop_times * single_loop_size + tail_num // thread_num
                    self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub, offset, n1_begin, n1_size2, m)
    return Oi_l1_zN


work = Path(tempfile.mkdtemp(prefix="h3_round49_broadcast_unroll_"))
os.chdir(work)
os.environ["FA_BR"] = "128"
os.environ["FA_BC"] = "304"
b, h, n, d = 1, 7, 63232, 128
nz = [b, h, d // 16, n // 16, 16, 16]
def desc(ori, dt):
    return {"shape": nz[:], "ori_shape": ori[:], "dtype": dt, "format": "NZ"}
name = "fa49_c30_broadcast_unroll"
instance = FlashAttentionFwd(q=desc([b,h,n,d], "float16"), k=desc([b,h,n,d], "float16"),
                            v=desc([b,h,n,d], "float16"), attn_mask=None, dropout_mask=None, alibi_mask=None,
                            kernel_name=name, tiling_stgy=TilingStrategy.from_strategy_name("sparse"), disable_debug=True)
instance.core_num = 30
instance.output_compute = types.MethodType(output_compute_unrolled, instance)
print("Building private broadcast-unroll variant at", work, flush=True)
instance.compute_process()
new_obj = work / "kernel_meta" / (name + ".o")
new_meta = json.loads(new_obj.with_suffix(".json").read_text())
assert new_meta["blockDim"] == 30
import torch
import torch_npu
torch.npu.set_device(0)
spec = importlib.util.spec_from_file_location("fa49_baseline", "/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py")
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
old_obj, old_name, old_bd = fa._build(b,h,n,d)
assert old_bd == 30
lib = fa._lib()
torch.manual_seed(49)
q, k, v = [torch.randn(nz, dtype=torch.float16, device="npu") * .05 for _ in range(3)]
l = torch.empty((b,h,n), dtype=torch.float32, device="npu")
outputs = {x: torch.empty_like(q) for x in ("baseline", "unroll")}
st = C.c_void_p(torch.npu.current_stream().npu_stream)
funcs = {}
for variant, obj, kernel in [("baseline", old_obj, old_name), ("unroll", str(new_obj), new_meta["kernelName"])]:
    handle = lib.fa910a_register(str(obj).encode(), kernel.encode())
    assert handle
    def call(variant=variant, handle=handle, kernel=kernel):
        rc = lib.fa910a_launch(handle, kernel.encode(), 30, q.data_ptr(), k.data_ptr(), v.data_ptr(),
                              outputs[variant].data_ptr(), l.data_ptr(), st)
        if rc:
            raise RuntimeError(f"launch rc={rc}")
    funcs[variant] = call
result = {"directory": str(work), "candidate_object": str(new_obj), "candidate_bytes": new_obj.stat().st_size,
          "baseline_object": old_obj, "runs": []}
for order in (("baseline", "unroll"), ("unroll", "baseline")):
    for variant in order:
        for _ in range(2):
            funcs[variant]()
        torch.npu.synchronize()
        ms = []
        for _ in range(5):
            start = time.perf_counter()
            for __ in range(3):
                funcs[variant]()
            torch.npu.synchronize()
            ms.append((time.perf_counter() - start) * 1000 / 3)
        row = {"variant": variant, "groups_ms": ms, "median_ms": statistics.median(ms)}
        result["runs"].append(row)
        print(json.dumps(row), flush=True)
result["bit_exact"] = bool(torch.equal(outputs["baseline"], outputs["unroll"]))
result["max_abs_diff"] = (outputs["baseline"].float() - outputs["unroll"].float()).abs().max().item()
(work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
