"""Peak device HBM bandwidth on one 910A.

The flash kernel re-reads K and V once per q-block: with Br=64 and N=63232 there
are 988 q-blocks, each sweeping all of K and V (2 x 113 MB), i.e. ~223 GB of
traffic per block per launch. At 672 ms that implies ~332 GB/s.

Whether that is near the hardware limit decides the next optimisation:
  * if peak is ~400 GB/s, the kernel is bandwidth-bound and the lever is raising
    Br (more q rows per K/V sweep), which needs an fp16 score tile to fit in UB;
  * if peak is ~1.2 TB/s, it is not bandwidth-bound and the lever is pipelining.
"""
import time

import torch
import torch_npu  # noqa: F401


def bench(fn, iters=5, warm=2):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    n = 256 * 1024 * 1024  # 512 MB fp16 buffer
    a = torch.empty(n, dtype=torch.float16, device="npu")
    b = torch.empty(n, dtype=torch.float16, device="npu")
    a.normal_()

    nbytes = a.numel() * 2
    # copy: read nbytes + write nbytes
    sec = bench(lambda: b.copy_(a))
    print("device-to-device copy : %7.2f ms -> %7.1f GB/s (rw)"
          % (sec * 1e3, 2 * nbytes / sec / 1e9))

    # read-only reduction-ish op
    sec = bench(lambda: a.sum())
    print("sum (read only)       : %7.2f ms -> %7.1f GB/s (r)"
          % (sec * 1e3, nbytes / sec / 1e9))

    print()
    print("kernel implied traffic : %.0f GB per launch (988 q-sweeps of K+V)"
          % (988 * 2 * 7 * 63232 * 128 * 2 / 1e9))
    print("at 672 ms              : %.0f GB/s" % (988 * 2 * 7 * 63232 * 128 * 2 / 0.672 / 1e9))


if __name__ == "__main__":
    main()