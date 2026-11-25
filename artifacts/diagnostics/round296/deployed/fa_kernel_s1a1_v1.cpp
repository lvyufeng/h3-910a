#include "kernel_operator.h"
using namespace AscendC;

// S1a-1: S = q(64x128) @ k(512x128)^T, FP32 accumulate, half output. Single core.
constexpr int32_t kM = 64;
constexpr int32_t kN = 512;
constexpr int32_t kK = 128;

extern "C" __global__ __aicore__ void flash_attention_custom(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o, GM_ADDR workspace, GM_ADDR tiling) {
    if (GetBlockIdx() != 0) { return; }
    TPipe pipe;
    TQue<TPosition::A1, 1> qL1;
    TQue<TPosition::B1, 1> kL1;
    TQue<TPosition::A2, 1> qL0A;
    TQue<TPosition::B2, 1> kL0B;
    TQue<TPosition::CO1, 1> cL0C;
    TQue<TPosition::VECOUT, 1> oUb;
    pipe.InitBuffer(qL1, 1, kM * kK * 2);
    pipe.InitBuffer(kL1, 1, kN * kK * 2);
    pipe.InitBuffer(qL0A, 1, kM * kK * 2);
    pipe.InitBuffer(kL0B, 1, kN * kK * 2);
    pipe.InitBuffer(cL0C, 1, kM * kN * 4);
    pipe.InitBuffer(oUb, 1, kM * kN * 2);

    GlobalTensor<half> qGm, kGm, oGm;
    qGm.SetGlobalBuffer((__gm__ half *)q, kM * kK);
    kGm.SetGlobalBuffer((__gm__ half *)k, kN * kK);
    oGm.SetGlobalBuffer((__gm__ half *)o, kM * kN);

    LocalTensor<half> aL1 = qL1.AllocTensor<half>();
    LocalTensor<half> bL1 = kL1.AllocTensor<half>();
    Nd2NzParams nd{};
    nd.ndNum = 1;
    nd.nValue = kM;
    nd.dValue = kK;
    nd.srcNdMatrixStride = 1;
    nd.srcDValue = kK;
    nd.dstNzC0Stride = kM;
    nd.dstNzNStride = 1;
    nd.dstNzMatrixStride = 0;
    DataCopy(aL1, qGm, nd);
    nd.nValue = kN;
    DataCopy(bL1, kGm, nd);
    qL1.EnQue(aL1);
    kL1.EnQue(bL1);
    LocalTensor<half> aL1r = qL1.DeQue<half>();
    LocalTensor<half> bL1r = kL1.DeQue<half>();

    LocalTensor<half> aL0 = qL0A.AllocTensor<half>();
    LocalTensor<half> bL0 = kL0B.AllocTensor<half>();
    LoadData2dParams ld{};
    ld.startIndex = 0;
    ld.srcStride = 1;
    ld.dstGap = 0;
    ld.ifTranspose = false;
    ld.repeatTimes = kM / 16;
    LoadData(aL0, aL1r, ld);
    ld.repeatTimes = kN / 16;
    ld.ifTranspose = true;
    LoadData(bL0, bL1r, ld);
    qL0A.EnQue(aL0);
    kL0B.EnQue(bL0);
    LocalTensor<half> aL0r = qL0A.DeQue<half>();
    LocalTensor<half> bL0r = kL0B.DeQue<half>();

    LocalTensor<float> cL0 = cL0C.AllocTensor<float>();
    MmadParams mp{};
    mp.m = kM;
    mp.n = kN;
    mp.k = kK;
    mp.cmatrixInitVal = true;
    mp.unitFlag = 0;
    Mmad(cL0, aL0r, bL0r, mp);
    cL0C.EnQue(cL0);
    LocalTensor<float> cL0r = cL0C.DeQue<float>();

    LocalTensor<half> oUbT = oUb.AllocTensor<half>();
    DataCopy(oUbT, cL0r, kM * kN);
    Cast(oUbT, cL0r, RoundMode::CAST_NONE, kM * kN);
    oUb.EnQue(oUbT);
    LocalTensor<half> oUbR = oUb.DeQue<half>();
    DataCopy(oGm, oUbR, kM * kN);
    oUb.FreeTensor(oUbR);
}
