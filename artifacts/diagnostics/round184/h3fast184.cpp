// R184: a flat C entry point so the audit hook can call the H3RangeStats custom op from Python
// through ctypes. One call per audited tensor; the caller takes the maximum magnitude bits and
// applies the same verdict rule the audit already uses.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_h3_range_stats.h"
#include <cstdint>
#include <vector>

namespace {
constexpr int64_t H3_CORE_NUM = 8;
constexpr int64_t H3_OUT_STRIDE = 8;
constexpr int64_t H3_OUT_ELEMS = H3_CORE_NUM * H3_OUT_STRIDE;
}  // namespace

extern "C" {

// Returns the largest FP16 magnitude bit field (0x0000..0x7fff) in the tensor, or -1 on error.
// devPtr must point to a contiguous float16 buffer already on the device.
int h3_max_mag_bits(const void *devPtr, int64_t rows, int64_t cols, void *streamVoid)
{
    aclrtStream stream = reinterpret_cast<aclrtStream>(streamVoid);
    int64_t dims[2] = {rows, cols};
    int64_t strides[2] = {cols, 1};
    // const_cast: the aclnn API takes a non-const handle but does not modify the input
    aclTensor *x = aclCreateTensor(dims, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 2,
                                   const_cast<void *>(devPtr));
    if (x == nullptr) { return -1; }

    static void *outDev = nullptr;
    static size_t outBytes = 0;
    if (outDev == nullptr) {
        outBytes = (size_t)H3_OUT_ELEMS * sizeof(int32_t);
        if (aclrtMalloc(&outDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            aclDestroyTensor(x);
            return -1;
        }
    }
    int64_t outDim = H3_OUT_ELEMS;
    int64_t outStride = 1;
    aclTensor *y = aclCreateTensor(&outDim, 1, ACL_INT32, &outStride, 0, ACL_FORMAT_ND,
                                   &outDim, 1, outDev);
    if (y == nullptr) { aclDestroyTensor(x); return -1; }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    int rc = (int)aclnnH3RangeStatsGetWorkspaceSize(x, y, &wsSize, &exec);
    if (rc != 0) { aclDestroyTensor(x); aclDestroyTensor(y); return -1; }
    void *ws = nullptr;
    if (wsSize > 0 && aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        aclDestroyTensor(x); aclDestroyTensor(y); return -1;
    }
    rc = (int)aclnnH3RangeStats(ws, wsSize, exec, stream);
    if (rc != 0) {
        if (ws) aclrtFree(ws);
        aclDestroyTensor(x); aclDestroyTensor(y);
        return -1;
    }
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        if (ws) aclrtFree(ws);
        aclDestroyTensor(x); aclDestroyTensor(y);
        return -1;
    }
    std::vector<int32_t> host((size_t)H3_OUT_ELEMS, 0);
    if (aclrtMemcpy(host.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        if (ws) aclrtFree(ws);
        aclDestroyTensor(x); aclDestroyTensor(y);
        return -1;
    }
    int best = 0;
    for (int64_t i = 0; i < H3_OUT_ELEMS; i++) {
        if (host[(size_t)i] > best) { best = host[(size_t)i]; }
    }
    if (ws) aclrtFree(ws);
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    return best;
}

}  // extern "C"
