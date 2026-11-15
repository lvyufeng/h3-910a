#!/usr/bin/env python3
"""Read-only 910A core/frequency audit and isolated FP16 throughput benchmarks.

Run only on a fleet-leased idle host. Does not change clocks, power limits, kernel
source, the production cache, or model files. Both FMA operations count as FLOPs.
The FA path is the deployed raw kernel: FP16 inputs, FP32 L0C accumulation. All
elapsed timings are bounded by synchronizing a single otherwise-idle device.
Telemetry is collected independently with DCMI getters, not torch operations.
"""
from __future__ import annotations

import argparse
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


class AICoreInfo(C.Structure):
    _fields_ = [("freq", C.c_uint), ("cur_freq", C.c_uint)]


class ChipInfo(C.Structure):
    _fields_ = [("chip_type", C.c_char * 32), ("chip_name", C.c_char * 32),
                ("chip_ver", C.c_char * 32), ("aicore_cnt", C.c_uint)]


class DCMI:
    def __init__(self, card=0):
        self.card = card
        self.lib = C.CDLL("/usr/local/Ascend/driver/lib64/driver/libdcmi.so")
        self.lib.dcmi_init.argtypes = []
        self.lib.dcmi_init.restype = C.c_int
        self.init_rc = self.lib.dcmi_init()
        for name, args in {
            "dcmi_get_device_aicore_info": [C.c_int, C.c_int, C.POINTER(AICoreInfo)],
            "dcmi_get_device_chip_info": [C.c_int, C.c_int, C.POINTER(ChipInfo)],
            "dcmi_get_device_frequency": [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_uint)],
            "dcmi_get_device_utilization_rate": [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_uint)],
            "dcmi_get_device_power_info": [C.c_int, C.c_int, C.POINTER(C.c_int)],
            "dcmi_get_device_temperature": [C.c_int, C.c_int, C.POINTER(C.c_int)],
            "dcmi_get_device_info": [C.c_int, C.c_int, C.c_int, C.c_uint, C.c_void_p, C.POINTER(C.c_uint)],
        }.items():
            f = getattr(self.lib, name)
            f.argtypes = args
            f.restype = C.c_int

    def scalar(self, name, *args, signed=False):
        value = C.c_int(0) if signed else C.c_uint(0)
        rc = getattr(self.lib, name)(self.card, 0, *args, C.byref(value))
        return {"rc": rc, "value": value.value if rc == 0 else None}

    def sample(self):
        ai = AICoreInfo()
        rc = self.lib.dcmi_get_device_aicore_info(self.card, 0, C.byref(ai))
        return {
            "time": time.time(),
            "aicore_info": {"rc": rc, "normal_mhz": ai.freq if not rc else None,
                            "current_mhz": ai.cur_freq if not rc else None},
            "aicore_current_mhz": self.scalar("dcmi_get_device_frequency", 7),
            "aicore_max_mhz": self.scalar("dcmi_get_device_frequency", 9),
            "hbm_mhz": self.scalar("dcmi_get_device_frequency", 6),
            "aicore_util_percent": self.scalar("dcmi_get_device_utilization_rate", 2),
            "hbm_bw_util_percent": self.scalar("dcmi_get_device_utilization_rate", 10),
            "power_raw": self.scalar("dcmi_get_device_power_info", signed=True),
            "temperature_c": self.scalar("dcmi_get_device_temperature", signed=True),
        }

    def static(self):
        chip = ChipInfo()
        rc = self.lib.dcmi_get_device_chip_info(self.card, 0, C.byref(chip))
        lp = {}
        # Header: DCMI_MAIN_CMD_LP=8, subcommand 9=frequency reduction cause,
        # 10=power info. Unsupported getters are recorded, never substituted.
        for sub, name in [(9, "frequency_reduction_cause"), (10, "rated_power_info")]:
            buf = (C.c_uint * 32)()
            size = C.c_uint(C.sizeof(buf))
            er = self.lib.dcmi_get_device_info(self.card, 0, 8, sub, buf, C.byref(size))
            lp[name] = {"rc": er, "bytes": size.value,
                        "words": list(buf)[:min(32, math.ceil(size.value / 4))] if er == 0 else None}
        cli = {}
        for kind in ("power", "health"):
            p = subprocess.run(["npu-smi", "info", "-t", kind, "-i", str(self.card), "-c", "0"],
                               capture_output=True, text=True, timeout=15)
            cli[kind] = {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
        return {"init_rc": self.init_rc, "chip_rc": rc,
                "chip_name": chip.chip_name.decode() if not rc else None,
                "physical_aicore_count": chip.aicore_cnt if not rc else None,
                "lp_getters": lp, "cli": cli, "idle": self.sample()}


class Sampler:
    def __init__(self, dcmi):
        self.dcmi = dcmi
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.samples.append(self.dcmi.sample())
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=10)

    def summary(self):
        result = {"samples": len(self.samples)}
        for key in ("aicore_current_mhz", "aicore_max_mhz", "hbm_mhz", "aicore_util_percent",
                    "hbm_bw_util_percent", "power_raw", "temperature_c"):
            values = [x[key]["value"] for x in self.samples if x[key]["rc"] == 0]
            result[key] = {"min": min(values), "max": max(values),
                           "median": statistics.median(values)} if values else {"unsupported": True}
        return result


def load_backend():
    path = "/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py"
    spec = importlib.util.spec_from_file_location("deployed_fa910a", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_kernel():
    import tbe.common.platform as plat
    plat.set_current_compile_soc_info("Ascend910A")
    sys.path.insert(0, "/tmp/fa_pkg")
    from flash_attention.ops.flash_attention.attention import FlashAttention
    import te.platform as teplat
    fa = load_backend()
    b, h, n, d = 1, 7, 63232, 128
    br, bc = fa.pick_tile(n, 128), fa.pick_tile(n, 304)
    obj = Path(fa._KERNEL_DIR) / f"b{b}_h{h}_n{n}_d{d}_br{br}_bc{bc}" / f"b{b}_h{h}_n{n}_d{d}_br{br}_bc{bc}.o"
    if not obj.is_file():
        raise RuntimeError("Audit refuses to rebuild or overwrite a missing production artifact")
    artifact, kernel, blocks = fa._build(b, h, n, d)
    mock = FlashAttention.__new__(FlashAttention) if not getattr(FlashAttention, "__abstractmethods__", ()) else None
    # Abstract base: instantiate its concrete subclass without invoking a build.
    if mock is None:
        from flash_attention.ops.flash_attention.flash_attention_fwd import FlashAttentionFwd
        mock = FlashAttentionFwd.__new__(FlashAttentionFwd)
    mock.core_num = plat.get_soc_spec(teplat.CORE_NUM)
    mock.B, mock.Tr, mock.Tc = b * h, n // br, n // bc
    mock.prev_block_num = mock.next_block_num = 65536
    mock.convert_py_dict_to_tik_tensor = lambda cm, ctm: (cm, ctm)
    cm, ctm = mock.get_core_task_info()
    counts = [sum(end - start for start, end in ctm[i].values()) for i in range(mock.core_num)]
    intervals = [{str(k): list(v) for k, v in ctm[i].items()} for i in range(mock.core_num)]
    # Coverage: every query tile is assigned once.
    coverage = [[0] * mock.Tr for _ in range(mock.B)]
    for it in intervals:
        for head, (lo, hi) in it.items():
            for index in range(lo, hi):
                coverage[int(head)][index] += 1
    assert all(c == 1 for hd in coverage for c in hd)
    with open(obj.with_suffix(".json")) as f:
        metadata = json.load(f)
    return {"artifact": artifact, "kernel": kernel, "block_dim": blocks,
            "soc": plat.get_soc_spec("SOC_VERSION"), "compile_core_count": mock.core_num,
            "Br": br, "Bc": bc, "Tr": mock.Tr, "Tc": mock.Tc,
            "query_tiles_total": mock.B * mock.Tr, "tiles_per_core": counts,
            "assignment": intervals, "complete_non_overlapping_coverage": True,
            "metadata": metadata}


def measure(torch, dcmi, label, fn, flops, dwell=3.0, iterations=12):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    groups = []
    with Sampler(dcmi) as sampler:
        for _ in range(5):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                fn()
            torch.npu.synchronize()
            groups.append((time.perf_counter() - t0) / iterations)
        deadline = time.monotonic() + dwell
        while time.monotonic() < deadline:
            for _ in range(12):
                fn()
            torch.npu.synchronize()
    sec = statistics.median(groups)
    data = {"label": label, "seconds": sec, "median_ms": sec * 1e3,
            "groups_ms": [x * 1e3 for x in groups],
            "tflops": flops / sec / 1e12 if flops else None,
            "telemetry": sampler.summary(), "telemetry_samples": sampler.samples}
    print(json.dumps({k: v for k, v in data.items() if k != "telemetry_samples"}), flush=True)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["inspect", "flash", "gemm", "copy", "all"], default="inspect")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dwell", type=float, default=3.0)
    args = ap.parse_args()
    result_path = Path(args.output)
    if result_path.exists():
        raise RuntimeError("Refuse to overwrite prior audit evidence")
    dcmi = DCMI(args.device)
    result = {"time": time.time(), "device": args.device, "static": dcmi.static(), "measurements": []}
    result["kernel"] = inspect_kernel()
    print(json.dumps({"static": result["static"], "kernel": result["kernel"]}), flush=True)
    if args.mode != "inspect":
        import torch
        import torch_npu
        torch.npu.set_device(args.device)
        torch.manual_seed(17)
        result["software"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "soc_version": torch_npu._C._npu_get_soc_version()}
        if args.mode in ("all", "gemm"):
            for m, k, n in [(11250, 5376, 28672), (4096, 5376, 8192), (8192, 8192, 8192),
                            (4096, 128, 8192), (8192, 128, 28672)]:
                a = torch.randn(m, k, dtype=torch.float16, device="npu") * 0.02
                b = torch.randn(k, n, dtype=torch.float16, device="npu") * 0.02
                out = torch.empty(m, n, dtype=torch.float16, device="npu")
                label = f"torch.mm FP16 M={m} K={k} N={n} ND input/output, preallocated"
                r = measure(torch, dcmi, label, lambda: torch.mm(a, b, out=out), 2 * m * k * n, args.dwell)
                ref = a[:8].float().cpu() @ b[:, :8].float().cpu()
                r["sample_max_abs_error"] = (out[:8, :8].float().cpu() - ref).abs().max().item()
                result["measurements"].append(r)
                del a, b, out
                torch.npu.empty_cache()
        if args.mode in ("all", "flash"):
            fa = load_backend()
            obj, kernel, blocks = fa._build(1, 7, 63232, 128)
            lib = fa._lib()
            handle = lib.fa910a_register(obj.encode(), kernel.encode())
            if not handle:
                raise RuntimeError("FA artifact registration failed")
            shape = (1, 7, 8, 63232 // 16, 16, 16)
            q, k, v = [torch.randn(shape, dtype=torch.float16, device="npu") * 0.05 for _ in range(3)]
            o = torch.empty_like(q)
            l = torch.empty((1, 7, 63232), dtype=torch.float32, device="npu")
            stream = C.c_void_p(torch.npu.current_stream().npu_stream)
            def launch():
                rc = lib.fa910a_launch(handle, kernel.encode(), blocks,
                                      q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), l.data_ptr(), stream)
                if rc != 0:
                    raise RuntimeError(f"FA launch failed, rc={rc}")
            result["measurements"].append(measure(torch, dcmi, "deployed FA raw FP16/FP32-accumulator, blockDim=32",
                                                  launch, 4 * 7 * 63232**2 * 128, args.dwell, iterations=3))
            del q, k, v, o, l
            torch.npu.empty_cache()
        if args.mode in ("all", "copy"):
            src = torch.zeros(256 * 1024**2, dtype=torch.float16, device="npu")
            dst = torch.empty_like(src)
            r = measure(torch, dcmi, "512 MiB device copy (read+write)", lambda: dst.copy_(src), 0, args.dwell)
            r["gbps_read_plus_write"] = 2 * src.numel() * src.element_size() / r["seconds"] / 1e9
            result["measurements"].append(r)
    result["after"] = dcmi.sample()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved read-only audit evidence:", result_path, flush=True)


if __name__ == "__main__":
    main()
