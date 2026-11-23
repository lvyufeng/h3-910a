#include "kernel_operator.h"
#include "h3_range_stats_tiling.h"

using namespace AscendC;

namespace {
constexpr int TILE_ELEMS = 16384;
constexpr int RED_WORK_ELEMS = TILE_ELEMS / 2 + 64;
constexpr unsigned int ALIGN_ELEMS = 128;

// Single pass over an FP16 tensor computing the largest magnitude.
//
// Abs() clears the sign bit, so the result carries exactly the FP16 magnitude bits
// 0x0000..0x7fff, which order monotonically with |x|. The host then reads:
//   maxMag >= 0x7c00  <=>  a NaN or Inf is present
//   maxMag == 0x7bff  <=>  some |x| == 65504 (largest finite magnitude)
//
// Why not the integer route: R182 measured that Ands() (the natural way to mask the sign
// bit) has no linkable instantiation for half, int16_t or uint16_t on this toolkit, while
// ReduceMax() supports only half and float. Abs() is a half op, so this pairing links.
//
// NaN caveat, deliberately left to the acceptance suite to measure: if ReduceMax() does not
// propagate NaN, a NaN-only tile would report a finite magnitude. R165 measured that claim
// for torch's max(); it has NOT been measured for the AscendC ReduceMax primitive, and the
// R180 adversarial cases ("nan_and_saturation", "many_bad") are exactly what decides it.
class H3RangeStatsKernel {
public:
    __aicore__ inline H3RangeStatsKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const H3RangeStatsTilingData &t)
    {
        unsigned int size = t.size;
        unsigned int coreNum = t.coreNum;
        unsigned int coreIdx = GetBlockIdx();
        if (coreNum == 0) { coreNum = 1; }
        unsigned int perCore = (size + coreNum - 1) / coreNum;
        perCore = (perCore + ALIGN_ELEMS - 1) / ALIGN_ELEMS * ALIGN_ELEMS;
        unsigned int start = coreIdx * perCore;
        unsigned int len = 0;
        if (start < size) {
            len = size - start;
            if (len > perCore) { len = perCore; }
        }
        len_ = len;
        xGm_.SetGlobalBuffer((__gm__ half *)x + start, len_);
        yGm_.SetGlobalBuffer((__gm__ int *)y + coreIdx, 1);
        pipe_.InitBuffer(inQueue_, 1, TILE_ELEMS * (int)sizeof(half));
        pipe_.InitBuffer(workQueue_, 1, RED_WORK_ELEMS * (int)sizeof(half));
        pipe_.InitBuffer(redQueue_, 1, 256);
        pipe_.InitBuffer(outQueue_, 1, 256);
    }

    __aicore__ inline void Process()
    {
        int blockMax = 0;
        unsigned int offset = 0;
        while (offset < len_) {
            unsigned int n = len_ - offset;
            if (n > (unsigned int)TILE_ELEMS) { n = (unsigned int)TILE_ELEMS; }
            CopyIn(offset, n);
            LocalTensor<half> x = inQueue_.DeQue<half>();
            Abs(x, x, (int32_t)n);                 // clear the sign bit in place
            int tileMax = TileMax(x, n);
            if (tileMax > blockMax) { blockMax = tileMax; }
            inQueue_.FreeTensor(x);
            offset += n;
        }
        LocalTensor<int> o = outQueue_.AllocTensor<int>();
        o.SetValue(0, blockMax);
        outQueue_.EnQue(o);
        LocalTensor<int> oc = outQueue_.DeQue<int>();
        DataCopy(yGm_, oc, 8);
        outQueue_.FreeTensor(oc);
    }

private:
    __aicore__ inline void CopyIn(unsigned int offset, unsigned int n)
    {
        LocalTensor<half> x = inQueue_.AllocTensor<half>();
        DataCopy(x, xGm_[offset], n);
        inQueue_.EnQue(x);
    }

    __aicore__ inline int TileMax(const LocalTensor<half> &x, unsigned int n)
    {
        LocalTensor<half> dst = redQueue_.AllocTensor<half>();
        LocalTensor<half> work = workQueue_.AllocTensor<half>();
        ReduceMax(dst, x, work, (int32_t)n, false);
        // the reduced value is produced on the vector pipe but GetValue reads it through the
        // scalar path, so the pipes must be synchronised or the read returns stale UB data
        // (that is exactly the garbage maxMag the first run produced)
        pipe_barrier(PIPE_ALL);
        half h = dst.GetValue(0);
        int v = (int)((*(unsigned short *)&h) & 0x7fff);
        workQueue_.FreeTensor(work);
        redQueue_.FreeTensor(dst);
        return v;
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECCALC, 1> workQueue_;
    TQue<QuePosition::VECCALC, 1> redQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<half> xGm_;
    GlobalTensor<int> yGm_;
    unsigned int len_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void h3_range_stats(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(H3RangeStatsTilingData);
    GET_TILING_DATA(tilingData, tiling);
    H3RangeStatsKernel op;
    op.Init(x, y, tilingData);
    op.Process();
}
