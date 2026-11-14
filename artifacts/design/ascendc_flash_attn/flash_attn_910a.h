// AscendC flash attention (forward, non-causal) for MiniMax-H3 on Ascend 910A.
//
// Ported from flash_attention-1.0.0 (TIK/TBE), see DESIGN.md for the contract
// this reproduces and why the win is avoiding the score materialisation.
//
// One AI core owns a set of Q blocks. Per Q block:
//   Qi (Br x D) -> L1 -> L0A
//   for j in KV blocks:
//     Kj (Bc x D) -> L1 -> L0B, Mmad -> Sij in L0C          (Br x Bc)
//     Fixpipe L0C -> UB as fp32
//     mi_new = max(mi, rowmax(Sij));  scale = exp(mi - mi_new)
//     Pij = exp(Sij - mi_new) -> fp16 -> L1
//     li  = li * scale + rowsum(Pij)
//     Vj -> L1 -> L0B, Mmad Pij*Vj -> L0C -> UB
//     Oi  = Oi * scale + Oij                                 (fp32, in UB)
//   O = Oi / li -> GM
//
// The cube/vector skew from the reference is kept: Sij for block j+1 is issued
// before block j's epilogue so the units overlap.

#ifndef FLASH_ATTN_910A_H
#define FLASH_ATTN_910A_H

#include "kernel_operator.h"

using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TPosition;
using AscendC::DataCopyParams;
using AscendC::DataCopyExtParams;

namespace flash_attn_910a {

constexpr int32_t kBlockNum = 16;              // fractal N0
constexpr int32_t kBr = 128;                   // Q rows per block
constexpr int32_t kBc = 128;                   // KV cols per block
// L1 byte budget on 910A is 1 MiB per core; Qi+Kj+Vj+Pij at 128x128 fp16 is
// 4 x 32 KiB = 128 KiB, leaving room for the double buffer of Kj.
constexpr int32_t kL1QueueDepth = 2;
// UB: Sij fp32 (Br x Bc) 64 KiB + Oi fp32 (Br x D) 64 KiB + reductions.
constexpr int32_t kUbQueueDepth = 1;

struct Tiling {
    int32_t seq_q;        // Nq
    int32_t seq_k;        // N
    int32_t head_dim;     // d, must be a multiple of 16
    int32_t num_heads;    // H
    int32_t br;
    int32_t bc;
    int32_t tr;           // ceil(Nq / br)
    int32_t tc;           // ceil(N / bc)
    int32_t blocks_per_core;
    float scale;          // 1/sqrt(d) folded into Q on load
};

__aicore__ inline int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

template <typename T>
class FlashAttn {
public:
    __aicore__ inline FlashAttn() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o,
                                const Tiling &t, TPipe *pipe) {
        tiling_ = t;
        pipe_ = pipe;
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v));
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(o));

        // L1 tiles. Pij is produced by the vector unit and consumed by the cube,
        // so it lives in L1 exactly as in the reference.
        pipe_->InitBuffer(qL1_, tiling_.br * tiling_.head_dim * sizeof(T));
        pipe_->InitBuffer(kL1_, tiling_.bc * tiling_.head_dim * sizeof(T) * kL1QueueDepth);
        pipe_->InitBuffer(vL1_, tiling_.bc * tiling_.head_dim * sizeof(T) * kL1QueueDepth);
        pipe_->InitBuffer(pL1_, tiling_.br * tiling_.bc * sizeof(T));

        // UB: the score tile and the running output stay fp32.
        pipe_->InitBuffer(sijUb_, tiling_.br * tiling_.bc * sizeof(float));
        pipe_->InitBuffer(oiUb_, tiling_.br * tiling_.head_dim * sizeof(float));
        pipe_->InitBuffer(rowUb_, tiling_.br * sizeof(float) * 3);  // mi, li, scale
    }

    __aicore__ inline void Process() {
        const int32_t coreId = AscendC::GetBlockIdx();
        const int32_t first = coreId * tiling_.blocks_per_core;
        int32_t last = first + tiling_.blocks_per_core;
        if (last > tiling_.tr) last = tiling_.tr;
        for (int32_t qb = first; qb < last; ++qb) {
            ComputeOneQBlock(qb);
        }
    }

private:
    // Load Qi, folding the softmax scale in on the way so the score tile never
    // needs a separate scaling pass (same reasoning as the PyTorch backend).
    __aicore__ inline void LoadQ(int32_t qb, LocalTensor<T> &qL1) {
        const int32_t rows = tiling_.br;
        const int64_t base = static_cast<int64_t>(qb) * rows * tiling_.head_dim;
        // Br x D contiguous rows: one burst per row group of 16.
        DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(rows);
        p.blockLen = static_cast<uint16_t>(tiling_.head_dim * sizeof(T) / 32);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(qL1, qGm_[base], p);
    }

    __aicore__ inline void ComputeOneQBlock(int32_t qb) {
        LocalTensor<T> qL1 = qL1_.Get<T>();
        LocalTensor<float> sij = sijUb_.Get<float>();
        LocalTensor<float> oi = oiUb_.Get<float>();
        LocalTensor<float> row = rowUb_.Get<float>();
        LocalTensor<float> mi = row;                              // [0, br)
        LocalTensor<float> li = row[tiling_.br];                  // [br, 2br)
        LocalTensor<float> sc = row[tiling_.br * 2];              // [2br, 3br)

        LoadQ(qb, qL1);
        AscendC::Duplicate(oi, 0.0f, tiling_.br * tiling_.head_dim);
        AscendC::Duplicate(mi, -3.0e38f, tiling_.br);
        AscendC::Duplicate(li, 0.0f, tiling_.br);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (int32_t kb = 0; kb < tiling_.tc; ++kb) {
            // --- S = Q @ K^T -------------------------------------------------
            LocalTensor<T> kL1 = kL1_.Get<T>();
            LoadKBlock(kb, kL1);
            LocalTensor<T> vL1 = vL1_.Get<T>();
            LoadVBlock(kb, vL1);

            LocalTensor<float> pij = sij;
            MatmulQK(qL1, kL1, pij);                              // Br x Bc fp32

            // --- online softmax ---------------------------------------------
            RowMax(pij, mi, sc);          // sc = exp(mi_old - mi_new), mi updated
            ExpSub(pij, mi);              // pij = exp(S - mi_new)
            RowSum(pij, li, sc);          // li = li * sc + rowsum

            // --- O = O * scale + P @ V --------------------------------------
            LocalTensor<T> pL1 = pL1_.Get<T>();
            CastToFp16(pL1, pij);
            MatmulPV(pL1, vL1, oi, sc);   // oi = oi * sc + P @ V
        }

        // --- normalise and store --------------------------------------------
        RecAndScale(oi, li);
        StoreO(qb, oi);
    }

    __aicore__ inline void LoadKBlock(int32_t kb, LocalTensor<T> &dst) {
        const int64_t base = static_cast<int64_t>(kb) * tiling_.bc * tiling_.head_dim;
        DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(tiling_.bc);
        p.blockLen = static_cast<uint16_t>(tiling_.head_dim * sizeof(T) / 32);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, kGm_[base], p);
    }

    __aicore__ inline void LoadVBlock(int32_t kb, LocalTensor<T> &dst) {
        const int64_t base = static_cast<int64_t>(kb) * tiling_.bc * tiling_.head_dim;
        DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(tiling_.bc);
        p.blockLen = static_cast<uint16_t>(tiling_.head_dim * sizeof(T) / 32);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(dst, vGm_[base], p);
    }

    // --- cube pieces ---------------------------------------------------------
    // These wrap LoadData + Mmad + Fixpipe. Declared here so the tiling of
    // L0A/L0B stays in one place; see the .cpp for the L0 buffers.
    __aicore__ inline void MatmulQK(const LocalTensor<T> &q, const LocalTensor<T> &k,
                                    LocalTensor<float> &out);
    __aicore__ inline void MatmulPV(const LocalTensor<T> &p, const LocalTensor<T> &v,
                                    LocalTensor<float> &oi, const LocalTensor<float> &scale);

    // --- vector pieces -------------------------------------------------------
    __aicore__ inline void RowMax(LocalTensor<float> &sij, LocalTensor<float> &mi,
                                  LocalTensor<float> &scale);
    __aicore__ inline void ExpSub(LocalTensor<float> &sij, const LocalTensor<float> &mi);
    __aicore__ inline void RowSum(const LocalTensor<float> &pij, LocalTensor<float> &li,
                                  const LocalTensor<float> &scale);
    __aicore__ inline void CastToFp16(LocalTensor<T> &dst, const LocalTensor<float> &src);
    __aicore__ inline void RecAndScale(LocalTensor<float> &oi, const LocalTensor<float> &li);
    __aicore__ inline void StoreO(int32_t qb, const LocalTensor<float> &oi);

    Tiling tiling_;
    TPipe *pipe_ = nullptr;
    GlobalTensor<T> qGm_, kGm_, vGm_, oGm_;
    TBuf<TPosition::A1> qL1_, kL1_, vL1_, pL1_;
    TBuf<TPosition::VECCALC> sijUb_, oiUb_, rowUb_;
};

}  // namespace flash_attn_910a

#endif  // FLASH_ATTN_910A_H
