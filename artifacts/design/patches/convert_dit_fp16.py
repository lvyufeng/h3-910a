"""Convert the H3 DiT checkpoint to fp16 so the DLO checkpoint-mmap path can use it.

The mmap path requires checkpoint dtype == runtime dtype, and we run the DiT in
fp16 (bf16 GEMM is ~168x slower on 910A). F32 tensors stay F32 because the
runtime keeps modulation-related params in fp32.
"""
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/mseco/models/MiniMax-H3-4step"
DST = sys.argv[2] if len(sys.argv) > 2 else SRC + "-fp16"
sub = "FL2VA/transformer"
src_dir = os.path.join(SRC, sub)
dst_dir = os.path.join(DST, sub)
os.makedirs(dst_dir, exist_ok=True)

files = sorted(os.listdir(src_dir))
for i, fn in enumerate(files, 1):
    outp = os.path.join(dst_dir, fn)
    if os.path.exists(outp) and os.path.getsize(outp) == os.path.getsize(os.path.join(src_dir, fn)):
        print("skip (done): %s" % fn, flush=True)
        continue
    with safe_open(os.path.join(src_dir, fn), framework="pt", device="cpu") as h:
        out = {}
        for k in h.keys():
            t = h.get_tensor(k)
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)
            out[k] = t.contiguous()
    save_file(out, outp, metadata={"format": "pt"})
    print("converted %d/%d: %s" % (i, len(files), fn), flush=True)
    del out

print("DIT CONVERSION DONE", flush=True)
