#!/usr/bin/env python3
"""R280: single-buffer FA timing at env-driven tiles, via the trim module that honours the caps."""
import json, os, statistics, sys
from pathlib import Path
import torch, torch_npu  # noqa: F401
from vllm_omni.diffusion.attention.backends import fa910a as FA
sys.path.insert(0, "/tmp/h3_budget149")
import bb218trim
FA.flash_attention_910a = bb218trim.flash_attention_910a

cuda = torch.npu
B, H, N, D = 1, 7, 41792, 128
torch.manual_seed(20261124)
q = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
k = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
v = (torch.randn(B, H, N, D, dtype=torch.float16, device="npu") * 0.05)
FA.flash_attention_910a(q, k, v)          # warm/compile
torch.npu.synchronize()
ts = []
for _ in range(4):
    torch.npu.synchronize()
    s, e = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    s.record(); out = FA.flash_attention_910a(q, k, v); e.record(); torch.npu.synchronize()
    ts.append(s.elapsed_time(e))
rep = {"schema": "r280.smalltile.v1",
       "tiles": [int(os.environ.get("H3_FA_BR_CAP", 128)), int(os.environ.get("H3_FA_BC_CAP", 304))],
       "median_ms": round(statistics.median(ts), 2), "all_ms": [round(t, 2) for t in ts],
       "finite": bool(torch.isfinite(out).all().item()), "sum": float(out.float().sum().item())}
Path(sys.argv[1]).write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(rep), flush=True)
