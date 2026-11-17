#!/usr/bin/env python3
"""Round82 GEMM prototype probe: build -> launch -> compare with an FP32 golden.

NOT EXECUTED by its author. The root runs this after reviewing the source; every
result it writes is device evidence, and none of it exists yet.

What it answers, in order
-------------------------
1. Does the TBE DSL emit an `f16f16f32` matmul for this SoC at all
   (`build_kernel` returns a kernel_meta pair)?
2. Does the legacy rtDevBinary launch run it without an AI Core exception?
3. Is the FP32 output bit-identical across two launches (repeatability), and how
   far is it from a float64/float32 numpy golden?
4. For the biased variant: is the result equal to `fp32(accum) + fp32(bias)`
   with a single FP16 cast, rather than the native Half-bias-add ordering?

What it deliberately does NOT answer
------------------------------------
* It is not the H3 geometry (M=2043, K=5120); that needs blocked tiling and is a
  separate, larger piece of work.
* A rounding match here does not by itself prove the full encoder agrees with the
  CPU FP16 reference. It only removes one identified error source.
* No speed claim: one micro kernel launch on one core is not an H3 measurement.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CANN = Path(os.environ.get("ASCEND_HOME_PATH", "/home/mseco/Ascend/cann-9.2.0"))


def compile_launcher(out_dir: Path) -> Path:
    """Build gemm_launcher82.c into a shared object with gcc."""
    src = HERE / "gemm_launcher82.c"
    so = out_dir / "libgemm82.so"
    include_candidates = [
        CANN / "include",
        CANN / "include/external",
        CANN / "runtime/include",
        CANN / "include/runtime",
    ]
    includes = [str(p) for p in include_candidates if p.is_dir()]
    cmd = ["gcc", "-O2", "-fPIC", "-shared", "-o", str(so), str(src)]
    for inc in includes:
        cmd += ["-I", inc]
    cmd += ["-L", str(CANN / "lib64"), "-lascendcl", "-lruntime", "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "launcher compile failed; gcc stderr follows. The include roots this "
            "install actually uses may need adjusting to the CANN layout:\n"
            + proc.stderr[-4000:])
    return so


def main() -> None:
    import argparse
    import numpy as np
    import torch
    import torch_npu  # noqa: F401  (registers the npu backend)

    import gemm_fp16_fp32out82 as proto

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="fp32_out", choices=sorted(proto.VARIANTS))
    parser.add_argument("--root", default="/tmp/h3_round82_gemm01")
    parser.add_argument("--m", type=int, default=proto.MICRO_M)
    parser.add_argument("--k", type=int, default=proto.MICRO_K)
    parser.add_argument("--n", type=int, default=proto.MICRO_N)
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"probe_{args.variant}.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence {receipt_path}")

    receipt = {
        "scope": __doc__,
        "variant": args.variant,
        "geometry": {"m": args.m, "k": args.k, "n": args.n},
        "executed": True,
        "not_h3_geometry": True,
        "not_a_speed_claim": True,
        "full_encoder_precision_accepted": False,
        "cases": {},
    }
    try:
        import hashlib
        receipt['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt['builder_sha256'] = hashlib.sha256((HERE/'gemm_fp16_fp32out82.py').read_bytes()).hexdigest()
        receipt['launcher_source_sha256'] = hashlib.sha256((HERE/'gemm_launcher82.c').read_bytes()).hexdigest()
        receipt['kernel_launch_executed'] = False
        capability = proto.capability_report()
        receipt["capability"] = capability

        build = proto.build_kernel(args.variant, args.m, args.k, args.n,
                                   out_dir=root / "kernels")
        receipt["build"] = build
        if not build.get("built"):
            raise RuntimeError("kernel build did not report built=true")

        so = compile_launcher(root)
        receipt["launcher"] = {"path": str(so), "compiled": True}

        lib = ctypes.CDLL(str(so))
        lib.gemm82_init.argtypes = [ctypes.c_int]
        lib.gemm82_init.restype = ctypes.c_int
        lib.gemm82_register.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        lib.gemm82_register.restype = ctypes.c_void_p
        for name in ("gemm82_launch3", "gemm82_launch4"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + \
                          ([ctypes.c_void_p] * (3 if name.endswith("3") else 4)) + \
                          [ctypes.c_void_p]
            fn.restype = ctypes.c_int
        lib.gemm82_sync.argtypes = [ctypes.c_void_p]
        lib.gemm82_sync.restype = ctypes.c_int
        torch.npu.set_device(0)
        context_probe = torch.empty((1,), dtype=torch.float16, device='npu:0')
        torch.npu.synchronize()
        del context_probe
        receipt['torch_context_initialized_before_registration'] = True

        if lib.gemm82_init(0) != 0:
            raise RuntimeError("gemm82_init failed")
        kernel_symbol = build["kernel_symbol"] or (build["kernel_name"] + "__kernel0")
        hdl = lib.gemm82_register(str(build["obj_path"]).encode(), kernel_symbol.encode())
        if not hdl:
            raise RuntimeError(f"gemm82_register returned NULL for {kernel_symbol}")
        receipt["registration"] = {"handle_nonnull": True, "kernel_symbol": kernel_symbol}

        use_bias = build["uses_fp32_bias"]
        m, k, n = args.m, args.k, args.n
        rng = np.random.default_rng(82082)
        a_np = (rng.standard_normal((m, k)).astype(np.float16))
        b_np = (rng.standard_normal((n, k)).astype(np.float16))
        # The bias parameter is originally Half just like the official CPUFP16
        # model;promoting that exact value to FP32 is only a kernel epilogue input.
        bias_half_np = (rng.standard_normal(n) * 0.1).astype(np.float16) if use_bias else None
        bias_np = bias_half_np.astype(np.float32) if use_bias else None

        # Golden: FP32 accumulate of the SAME FP16 operand values, bias in FP32,
        # then a single cast to the kernel's output dtype.
        golden = a_np.astype(np.float32) @ b_np.astype(np.float32).T
        if use_bias:
            golden = golden + bias_np[None, :]
        out_dtype = np.float16 if build["dst_dtype"] == "float16" else np.float32
        golden_out = golden.astype(out_dtype)

        dev = torch.device("npu", 0)
        a = torch.from_numpy(a_np).to(dev)
        b = torch.from_numpy(b_np).to(dev)
        bias = torch.from_numpy(bias_np).to(dev) if use_bias else None
        out = torch.zeros((m, n), dtype=torch.float16 if out_dtype is np.float16
                          else torch.float32, device=dev)
        stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        if not stream.value:
            raise RuntimeError("no current torch NPU stream")
        receipt['current_stream_source'] = 'torch.npu.current_stream().npu_stream'
        blocks = int(build.get("block_dim") or 1)

        def launch():
            if use_bias:
                rc = lib.gemm82_launch4(hdl, blocks, a.data_ptr(), b.data_ptr(),
                                        bias.data_ptr(), out.data_ptr(), stream)
            else:
                rc = lib.gemm82_launch3(hdl, blocks, a.data_ptr(), b.data_ptr(),
                                        out.data_ptr(), stream)
            if rc != 0:
                raise RuntimeError(f"rtKernelLaunch returned {rc}")
            rc = lib.gemm82_sync(stream)
            if rc != 0:
                raise RuntimeError(f"aclrtSynchronizeStream returned {rc}")

        torch.npu.synchronize()
        started = time.perf_counter()
        launch()
        receipt['kernel_launch_executed'] = True
        receipt["cases"]["first_launch_s_including_sync"] = time.perf_counter() - started
        first = out.detach().cpu().contiguous().numpy().copy()
        started = time.perf_counter()
        launch()
        second = out.detach().cpu().contiguous().numpy().copy()
        receipt["cases"]["second_launch_s_including_sync"] = time.perf_counter() - started

        def compare(got, want):
            got64, want64 = got.astype(np.float64), want.astype(np.float64)
            delta = got64 - want64
            return {
                "finite": bool(np.isfinite(got).all()),
                "raw_bits_equal": bool(np.array_equal(got.view(np.uint8),
                                                      want.view(np.uint8))),
                "max_abs_error": float(np.abs(delta).max()),
                "relative_rms": float(np.sqrt((delta ** 2).mean())
                                      / max(float(np.sqrt((want64 ** 2).mean())), 1e-30)),
                "elements": int(got.size),
            }

        receipt["cases"]["repeat_vs_first"] = compare(second, first)
        receipt["cases"]["vs_fp32_golden"] = compare(first, golden_out)
        # Actual original CPU Half operator,not a whole-FP32 teacher adapter.
        torch.set_num_threads(4)
        ac = torch.from_numpy(a_np);bc = torch.from_numpy(b_np)
        bh = torch.from_numpy(bias_half_np) if use_bias else None
        with torch.inference_mode():
            cpu_half = torch.nn.functional.linear(ac,bc,bh).numpy()
            native_half = torch.nn.functional.linear(a,b,bias.half() if use_bias else None).cpu().numpy()
        receipt['cases']['kernel_cast_to_half_vs_CPU_FP16_linear'] = compare(first.astype(np.float16),cpu_half)
        receipt['cases']['native_half_vs_CPU_FP16_linear'] = compare(native_half,cpu_half)
        receipt['CPU_FP16_reference_operator'] = 'torch.nn.functional.linear(Half input,Half weight,Half bias)'
        receipt['FP32_numpy_golden_is_only_accumulator_diagnostic'] = True
        if not receipt['cases']['repeat_vs_first']['raw_bits_equal'] or not receipt['cases']['vs_fp32_golden']['finite']:
            raise RuntimeError('kernel repeat or finite control failed')
        if use_bias:
            # Explicitly distinguish "FP32 bias add then one cast" from the
            # native path's extra Half rounding of the bias, using the SAME
            # operand values.
            native_like = (golden - bias_np[None, :]).astype(np.float16) \
                .astype(np.float32) + bias_np[None, :].astype(np.float16).astype(np.float32)
            receipt["cases"]["vs_native_half_bias_ordering"] = compare(
                first, native_like.astype(out_dtype))
            receipt["cases"]["bias_rounding_note"] = (
                "native torch Half Linear was measured (Round80) to equal a no-bias "
                "Half GEMM plus a Half bias add; this variant should instead match "
                "the FP32-bias golden")
        receipt["completed"] = True
    except BaseException as error:
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)[:4000]}
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print(json.dumps({"receipt": str(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()