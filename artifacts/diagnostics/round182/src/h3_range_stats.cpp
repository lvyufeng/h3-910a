#include "kernel_operator.h"
#include "h3_range_stats_tiling.h"

using namespace AscendC;

namespace {
constexpr int TILE_ELEMS = 16384;
constexpr int RED_WORK_ELEMS = TILE_ELEMS / 2 + 64;
constexpr unsigned int ALIGN_ELEMS = 128;

// Single pass over an FP16 tensor reducing the magnitude bit field.
// The FP16 magnitude bits 0x0000..0x7fff order monotonically with |x|, so
//   maxMag >= 0x7c00  <=>  a NaN or Inf is present
//   maxMag == 0x7bff  <=>  some |x| == 65504 (largest finite magnitude)
// Only bit patterns are used: R165 measured that this device cannot be trusted to
// report NaN through its FP16 float predicates.
//
// API constraints found from the real compile (R182):
//   Ands       supports uint16/int16/int64/uint64, NOT half
//   ReduceMax  supports half/float, NOT the integer types
//   -> mask the sign bit on an int16 view, then cast to float (exact for int16) and reduce
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
        pipe_.InitBuffer(fbufQueue_, 1, TILE_ELEMS * (int)sizeof(float));
        pipe_.InitBuffer(workQueue_, 1, RED_WORK_ELEMS * (int)sizeof(float));
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
            int tileMax = TileMagnitudeMax(x, n);
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

    __aicore__ inline int TileMagnitudeMax(const LocalTensor<half> &x, unsigned int n)
    {
        // 1. mask the sign bit; Ands needs an integer type, so view the same buffer as int16
        LocalTensor<uint16_t> bits = x.template ReinterpretCast<uint16_t>();
        Ands(bits, bits, (uint16_t)0x7fff, (int32_t)n);

        // 2. widen to float, which represents every int16 value exactly, so the
        //    saturation threshold 0x7bff and the non-finite threshold 0x7c00 stay distinct
        LocalTensor<float> fb = fbufQueue_.AllocTensor<float>();
        // uint16 magnitudes are all <= 32767, so the widening is exact
        LocalTensor<int16_t> asI16 = bits.template ReinterpretCast<int16_t>();
        Cast(fb, asI16, RoundMode::CAST_NONE, (int32_t)n);

        // 3. reduce in float, the only dtype ReduceMax accepts besides half
        LocalTensor<float> dst = redQueue_.AllocTensor<float>();
        LocalTensor<float> work = workQueue_.AllocTensor<float>();
        ReduceMax(dst, fb, work, (int32_t)n, false);
        int v = (int)dst.GetValue(0);
        workQueue_.FreeTensor(work);
        redQueue_.FreeTensor(dst);
        fbufQueue_.FreeTensor(fb);
        return v;
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECCALC, 1> fbufQueue_;
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
