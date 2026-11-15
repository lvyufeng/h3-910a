"""What does a matmul at the kernel's REAL per-tile shape achieve?

The kernel's inner matmuls are M=Br=64, K=128, N=Bc=608 (QK^T) and
M=64, K=608, N=128 (PV). Both M and the reduction are small, so cube
efficiency is likely far below even the 25 TFLOPS measured at M=4096, K=128.

Measured with a batch dimension so launch overhead does not dominate.

If small M is the limiter, then raising the effective M per K/V tile (process
several q-blocks against one loaded K/V) is a different and possibly cheaper
lever than pipelining.
"""
import time

import torch
import torch_npu  # noqa: F401

D = 128


def bench(fn, iters=3, warm=1):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    print("QK^T-like:  A(batch,M,128) @ B(batch,128,N)")
    print("%8s %6s %6s %6s %10s %10s" % ("batch", "M", "K", "N", "ms", "TFLOPS"))
    for batch, M, K, N in [
        (64, 64, D, 608),     # one q-block, as the kernel does it now
        (64, 128, D, 608),
        (64, 256, D, 608),    # several q-blocks sharing one K/V tile
        (64, 512, D, 608),
        (64, 64, D, 4096),
        (16, 64, D, 63232),   # a whole row of q-blocks vs the full K
    ]:
        a = torch.randn(batch, M, K, dtype=torch.float16, device="npu")
        b = torch.randn(batch, K, N, dtype=torch.float16, device="npu")
        sec = bench(lambda: torch.bmm(a, b))
        flops = 2 * batch * M * K * N
        print("%8d %6d %6d %6d %10.2f %10.1f"
              % (batch, M, K, N, sec * 1e3, flops / sec / 1e12), flush=True)
        del a, b


if __name__ == "__main__":
    main()