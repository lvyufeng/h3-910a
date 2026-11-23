// R182 acceptance test for the H3RangeStats custom op.
//
// Verifies the op against bit-pattern expectations derived independently from struct.pack
// (see round180/reference180.py), then times it on the three audited tensor shapes.
//
// The verdict rule under test:
//   maxMag >= 0x7c00  <=> a NaN or Inf is present
//   maxMag == 0x7bff  <=> some |x| == 65504 (largest finite magnitude)
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnn_h3_range_stats.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            printf("FAIL %s:%d %s\n", __FILE__, __LINE__, msg);               \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static constexpr uint16_t FP16_MAX_MAG = 0x7bff;
static constexpr uint16_t INF_NAN_BASE = 0x7c00;

struct Case {
    const char *name;
    std::vector<uint16_t> bits;   // raw fp16 patterns
    bool want_nonfinite;
    bool want_saturated;
};

static uint16_t f16(float v)
{
    // encode a float as fp16 bits without depending on _Float16 support
    if (std::isnan(v)) return 0x7e00;
    if (std::isinf(v)) return v > 0 ? 0x7c00 : 0xfc00;
    uint32_t f;
    std::memcpy(&f, &v, 4);
    uint32_t sign = (f >> 16) & 0x8000u;
    int32_t exp = (int32_t)((f >> 23) & 0xff) - 127;
    uint32_t mant = f & 0x7fffffu;
    if (exp > 15) return (uint16_t)(sign | 0x7c00u);
    if (exp < -24) return (uint16_t)sign;
    if (exp < -14) {
        uint32_t shift = (uint32_t)(-14 - exp);
        uint32_t m = (mant | 0x800000u) >> (13 + shift);
        return (uint16_t)(sign | m);
    }
    uint32_t h = ((uint32_t)(exp + 15) << 10) | (mant >> 13);
    return (uint16_t)(sign | h);
}

// run the op once; returns the device-computed max magnitude bits
static int run_op(const std::vector<uint16_t> &bits, int64_t rows, int64_t cols,
                  aclrtStream stream, double *ms_out)
{
    size_t n = bits.size();
    size_t bytes = n * sizeof(uint16_t);

    void *dIn = nullptr, *dOut = nullptr;
    aclrtMalloc(&dIn, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    const int64_t OUT_ELEMS = 8;    // one slot per core, matching H3_CORE_NUM in op_host
    size_t outBytes = (size_t)OUT_ELEMS * sizeof(int32_t);
    aclrtMalloc(&dOut, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dIn, bytes, bits.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    int64_t dims[2] = {rows, cols};
    int64_t strides[2] = {cols, 1};
    aclTensor *x = aclCreateTensor(dims, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, dims, 2, dIn);
    int64_t os = 1;
    aclTensor *y = aclCreateTensor(&OUT_ELEMS, 1, ACL_INT32, &os, 0, ACL_FORMAT_ND, &OUT_ELEMS, 1, dOut);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    auto st = aclnnH3RangeStatsGetWorkspaceSize(x, y, &wsSize, &exec);
    if (st != 0) { printf("GetWorkspaceSize failed %d\n", (int)st); return -1; }
    void *ws = nullptr;
    if (wsSize > 0) aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);

    auto t0 = std::chrono::steady_clock::now();
    st = aclnnH3RangeStats(ws, wsSize, exec, stream);
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::steady_clock::now();
    if (ms_out) *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (st != 0) { printf("op launch failed %d\n", (int)st); return -1; }

    std::vector<int32_t> outs((size_t)OUT_ELEMS, 0);
    aclrtMemcpy(outs.data(), outBytes, dOut, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    int32_t out = 0;
    for (int64_t i = 0; i < OUT_ELEMS; i++) if (outs[(size_t)i] > out) out = outs[(size_t)i];

    if (ws) aclrtFree(ws);
    aclDestroyTensor(x); aclDestroyTensor(y);
    aclrtFree(dIn); aclrtFree(dOut);
    return out;
}

int main()
{
    CHECK(aclInit(nullptr) == ACL_SUCCESS, "aclInit");
    CHECK(aclrtSetDevice(0) == ACL_SUCCESS, "setDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream) == ACL_SUCCESS, "createStream");

    std::vector<Case> cases = {
        {"all_finite_normal", {f16(1.0f), f16(-2.5f), f16(100.0f)}, false, false},
        {"single_nan", {f16(1.0f), 0x7e00, f16(3.0f)}, true, false},
        {"single_posinf", {f16(1.0f), 0x7c00, f16(3.0f)}, true, false},
        {"single_neginf", {f16(1.0f), 0xfc00, f16(3.0f)}, true, false},
        {"single_pos_saturation", {f16(1.0f), f16(65504.0f)}, false, true},
        {"single_neg_saturation", {f16(1.0f), f16(-65504.0f)}, false, true},
        {"both_signs", {f16(65504.0f), f16(-65504.0f)}, false, true},
        {"nan_and_saturation", {0x7e00, f16(65504.0f)}, true, false},
        {"inf_and_saturation", {0x7c00, f16(-65504.0f)}, true, false},
        {"many_bad", {0x7e00, 0x7c00, 0xfc00, f16(65504.0f), f16(-65504.0f), f16(1.0f)}, true, false},
        {"just_below_max", {f16(65472.0f), f16(-65472.0f)}, false, false},
        {"rounds_up_to_max", {f16(65503.0f), f16(-65503.0f)}, false, true},
        {"subnormals", {f16(6.1e-5f), f16(-6.1e-5f), f16(1e-7f)}, false, false},
        {"zeros", {0x0000, 0x8000}, false, false},
        {"max_finite_many", {0x7bff, 0x7bff, 0x7bff, 0x7bff}, false, true},
    };

    int failures = 0;
    for (auto &c : cases) {
        int maxMag = run_op(c.bits, 1, (int64_t)c.bits.size(), stream, nullptr);
        if (maxMag < 0) { printf("FAIL %-24s launch error\n", c.name); failures++; continue; }
        bool gotNf = maxMag >= (int)INF_NAN_BASE;
        bool gotSat = maxMag == (int)FP16_MAX_MAG;
        bool ok = (gotNf == c.want_nonfinite) && (gotSat == c.want_saturated);
        printf("%-24s maxMag=0x%04x nonfinite=%d(want %d) saturated=%d(want %d) %s\n",
               c.name, maxMag, (int)gotNf, (int)c.want_nonfinite,
               (int)gotSat, (int)c.want_saturated, ok ? "OK" : "MISMATCH");
        if (!ok) failures++;
    }
    printf("adversarial: %d/%zu passed\n", (int)cases.size() - failures, cases.size());

    // timing on the audited shapes
    struct Shape { const char *name; int64_t rows, cols; } shapes[] = {
        {"mlp_fc1(10448x14336)", 10448, 14336},
        {"qkv_proj(10448x10752)", 10448, 10752},
        {"adaln(3x96768)", 3, 96768},
    };
    for (auto &s : shapes) {
        std::vector<uint16_t> data((size_t)(s.rows * s.cols), 0x3c00);
        double best = 1e9;
        for (int i = 0; i < 8; i++) {
            double ms = 0;
            run_op(data, s.rows, s.cols, stream, &ms);
            if (i >= 3 && ms < best) best = ms;
        }
        printf("TIMING %-24s %.3f ms\n", s.name, best);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    printf(failures == 0 ? "RESULT PASS\n" : "RESULT FAIL\n");
    return failures == 0 ? 0 : 1;
}
