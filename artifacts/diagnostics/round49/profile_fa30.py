#!/usr/bin/env python3
"""Hardware pipe counters for the verified 30-core FA kernel, isolated single NPU.

No model bypasses or in-situ global timing. Allocation and kernel registration are
outside profiling. ExperimentalConfig Level1 is required for AI Core metrics.
This uses exactly the installed torch_npu profiler API inspected for this run.
"""
from __future__ import annotations
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys
import time
import torch
import torch_npu
from torch_npu import profiler as p

ROOT = Path("/tmp/h3_round49_pipe_profile01")
if ROOT.exists():
    raise RuntimeError("Refuse to overwrite earlier profiling data")
ROOT.mkdir()
torch.npu.set_device(0)
path = "/home/mseco/vllm-omni/vllm_omni/diffusion/attention/backends/fa910a.py"
spec = importlib.util.spec_from_file_location("fa_audit", path)
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)
obj, kernel, bd = fa._build(1, 7, 63232, 128)
assert bd == 30, (obj, bd)
lib = fa._lib()
handle = lib.fa910a_register(obj.encode(), kernel.encode())
assert handle, "registration failed"
shape = (1, 7, 8, 63232 // 16, 16, 16)
torch.manual_seed(29)
q, k, v = [torch.randn(shape, dtype=torch.float16, device="npu") * .05 for _ in range(3)]
o = torch.empty_like(q)
l = torch.empty((1, 7, 63232), dtype=torch.float32, device="npu")
st = C.c_void_p(torch.npu.current_stream().npu_stream)


def launch():
    rc = lib.fa910a_launch(handle, kernel.encode(), bd, q.data_ptr(), k.data_ptr(),
                          v.data_ptr(), o.data_ptr(), l.data_ptr(), st)
    if rc:
        raise RuntimeError(f"raw launch failed {rc}")


for _ in range(3):
    launch()
torch.npu.synchronize()
start = time.perf_counter()
for _ in range(3):
    launch()
torch.npu.synchronize()
unprofiled_ms = (time.perf_counter() - start) * 1000 / 3
cfg = p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1,
                            aic_metrics=p.AiCMetrics.PipeUtilization,
                            export_type=p.ExportType.Text)
with p.profile(activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
               record_shapes=False, profile_memory=False, with_stack=False,
               experimental_config=cfg,
               on_trace_ready=p.tensorboard_trace_handler(str(ROOT), worker_name="FA30", async_mode=False)) as prof:
    for _ in range(3):
        launch()
    torch.npu.synchronize()
result = {"output_root": str(ROOT), "object": obj, "kernel": kernel, "block_dim": bd,
          "unprofiled_ms": unprofiled_ms, "profiled_launches": 3,
          "metric": "PipeUtilization", "torch": torch.__version__, "torch_npu": torch_npu.__version__}
(ROOT / "invocation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
