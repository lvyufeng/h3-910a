#include "kernel_operator.h"
#include "flash_attention_custom_tiling.h"
using namespace AscendC;

constexpr int32_t SZ = 16;               // 16x16 matrices
constexpr int32_t ELEMS = SZ * SZ;

extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FlashAttentionCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (GetBlockIdx() != 0) { return; }
    TPipe pipe;
    TQue<TPosition::A1, 1> aL1;
    TQue<TPosition::B1, 1> bL1;
    TQue<TPosition::A2, 1> aL0;
    TQue<TPosition::B2, 1> bL0;
    TQue<TPosition::CO1, 1> cL0;
    pipe.InitBuffer(aL1, 1, ELEMS * 2);
    pipe.InitBuffer(bL1, 1, ELEMS * 2);
    pipe.InitBuffer(aL0, 1, ELEMS * 2);
    pipe.InitBuffer(bL0, 1, ELEMS * 2);
    pipe.InitBuffer(cL0, 1, ELEMS * 4);

    GlobalTensor<half> qGm, kGm, oGm;
    qGm.SetGlobalBuffer((__gm__ half *)q, ELEMS);
    kGm.SetGlobalBuffer((__gm__ half *)k, ELEMS);
    oGm.SetGlobalBuffer((__gm__ half *)o, ELEMS);

    // 16x16 ND equals NZ (single 16x16 fractal) -> plain linear copy is enough
    LocalTensor<half> a1 = aL1.AllocTensor<half>();
    LocalTensor<half> b1 = bL1.AllocTensor<half>();
    DataCopy(a1, qGm, ELEMS);
    DataCopy(b1, kGm, ELEMS);
    aL1.EnQue(a1); bL1.EnQue(b1);
    LocalTensor<half> a1r = aL1.DeQue<half>();
    LocalTensor<half> b1r = bL1.DeQue<half>();

    LocalTensor<half> a0 = aL0.AllocTensor<half>();
    LocalTensor<half> b0 = bL0.AllocTensor<half>();
    LoadData2dParams ld;
    ld.startIndex = 0;
    ld.repeatTimes = 1;      // one 16x16 fractal
    ld.srcStride = 1;
    ld.dstGap = 0;
    ld.ifTranspose = false;
    LoadData(a0, a1r, ld);
    LoadData(b0, b1r, ld);
    aL0.EnQue(a0); bL0.EnQue(b0);
    LocalTensor<half> a0r = aL0.DeQue<half>();
    LocalTensor<half> b0r = bL0.DeQue<half>();

    LocalTensor<float> c0 = cL0.AllocTensor<float>();
    MmadParams mp;
    mp.m = SZ; mp.n = SZ; mp.k = SZ;
    mp.cmatrixInitVal = true;
    Mmad(c0, a0r, b0r, mp);
    cL0.EnQue(c0);
    LocalTensor<float> c0r = cL0.DeQue<float>();

    TBuf<TPosition::VECCALC> fBuf, hBuf;
    pipe.InitBuffer(fBuf, ELEMS * 4);
    pipe.InitBuffer(hBuf, ELEMS * 2);
    LocalTensor<float> cf = fBuf.Get<float>();
    LocalTensor<half> ch = hBuf.Get<half>();
    DataCopyCO12DstParams cp;
    cp.nSize = SZ; cp.mSize = SZ;
    cp.dstStride = 0; cp.srcStride = 0;
    cp.quantPre = QuantMode_t::NoQuant;
    cp.reluPre = 0; cp.channelSplit = false; cp.nz2ndEn = false;
    DataCopy(cf, c0r, cp);
    PipeBarrier<PIPE_ALL>();
    Cast(ch, cf, RoundMode::CAST_NONE, ELEMS);
    PipeBarrier<PIPE_ALL>();
    DataCopy(oGm, ch, ELEMS);
}
