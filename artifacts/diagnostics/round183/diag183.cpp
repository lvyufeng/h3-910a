// R183: why are the verdicts wrong?
//
// The kernel masks every magnitude to <= 0x7fff, so it can never write a value above 0x7fff.
// The R182 run read back 0x3d443d5e-style values, which means the test was not reading what
// the kernel wrote. This probe distinguishes the two possibilities by pre-filling the output
// with a sentinel and printing every slot.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_h3_range_stats.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const int64_t OUT_ELEMS = 64;
static const uint32_t SENTINEL = 0xDEADBEEF;

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // input: 1024 elements of 1.0 (fp16 bits 0x3c00). Expected maxMag = 0x3c00.
    std::vector<uint16_t> host(1024, 0x3c00);   // 1024 is a multiple of 128
    size_t inBytes = host.size() * sizeof(uint16_t);
    size_t outBytes = OUT_ELEMS * sizeof(int32_t);

    void *dIn = nullptr, *dOut = nullptr;
    aclrtMalloc(&dIn, inBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dIn, inBytes, host.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int32_t> fill(OUT_ELEMS, (int32_t)SENTINEL);
    aclrtMemcpy(dOut, outBytes, fill.data(), outBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t dims[2] = {1, 1024};
    int64_t strides[2] = {1024, 1};
    aclTensor *x = aclCreateTensor(dims, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 2, dIn);
    int64_t os = 1;
    aclTensor *y = aclCreateTensor(&OUT_ELEMS, 1, ACL_INT32, &os, 0, ACL_FORMAT_ND, &OUT_ELEMS, 1, dOut);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    auto st = aclnnH3RangeStatsGetWorkspaceSize(x, y, &wsSize, &exec);
    printf("GetWorkspaceSize st=%d wsSize=%llu\n", (int)st, (unsigned long long)wsSize);
    if (st != 0) { printf("RESULT: cannot launch\n"); return 1; }
    void *ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
    st = aclnnH3RangeStats(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);
    printf("launch st=%d\n", (int)st);

    std::vector<int32_t> got(OUT_ELEMS, 0);
    aclrtMemcpy(got.data(), outBytes, dOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    bool anyWritten = false;
    printf("slots:");
    for (int i = 0; i < OUT_ELEMS; i++) {
        printf(" [%d]=0x%08x", i, (unsigned)got[(size_t)i]);
        if ((uint32_t)got[(size_t)i] != SENTINEL) anyWritten = true;
    }
    printf("\n");
    printf("expected maxMag 0x3c00 for an all-1.0 input\n");
    printf("kernel wrote anything: %s\n", anyWritten ? "YES" : "NO - the sentinel is untouched");
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
