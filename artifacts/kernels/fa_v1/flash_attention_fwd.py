from tbe import tik

from flash_attention.ops.flash_attention.attention import FlashAttention
from flash_attention.ops.flash_attention.constants import FP16
from flash_attention.ops.flash_attention.constants import FP32
from flash_attention.ops.flash_attention.constants import GM
from flash_attention.ops.flash_attention.constants import L1
from flash_attention.ops.flash_attention.constants import UB
from flash_attention.ops.flash_attention.tiling_strategy.strategy import TilingStrategy


class FlashAttentionFwd(FlashAttention):
    """The implementation of flash attention forward
    This function contains the flash attention forward implementation used in flash attention (see paper)
    `FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
    <https://arxiv.org/pdf/2307.08691.pdf>`
    """

    def __init__(self, q, k, v,
                 attn_mask, dropout_mask, alibi_mask,
                 kernel_name,
                 tiling_stgy: TilingStrategy,
                 prev_block_num=65536,
                 next_block_num=65536,
                 disable_debug=True):
        super().__init__(q, k, v, attn_mask, dropout_mask, alibi_mask, kernel_name,
                         tiling_stgy, prev_block_num, next_block_num, disable_debug)

    def define_custom_inputs(self):
        pass

    def define_outputs(self):
        """define output gm tensors"""
        self.O_gm = self.tik_instance.Tensor(FP16, self.O_shape, name="O_gm", scope=GM, is_atomic_add=True)
        self.l_gm = self.tik_instance.Tensor(FP32, self.l_shape, name="l_gm", scope=GM, is_atomic_add=True)

    def prepare_global_ones(self):
        """Prepare global ones tensor in L1 for cube impl row_sum"""
        Bc_aligned = (self.Bc + 15) // 16 * 16
        last_Bc_aligned = (self.last_Bc + 15) // 16 * 16
        self.ones_l1 = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_l1", scope=L1)
        self.last_ones_l1 = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_l1", scope=L1)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            ones_ub = self.tik_instance.Tensor(FP16, (Bc_aligned, 16), name="ones_ub", scope=UB)
            self.tik_instance.h_duplicate(ones_ub, 1.0)
            self.cont_data_mv_1_bust(dst=self.ones_l1, src=ones_ub, burst=Bc_aligned)
            last_ones_ub = self.tik_instance.Tensor(FP16, (last_Bc_aligned, 16), name="last_ones_ub", scope=UB)
            self.tik_instance.h_duplicate(ones_ub, 1.0)
            self.cont_data_mv_1_bust(dst=self.last_ones_l1, src=last_ones_ub, burst=last_Bc_aligned)

    def rowmax_compute(self, Sij_ub_zN, mi_ub, kv_blk_idx):
        """Calculate row max for softmax of flashattention2
        :param Sij_ub_zN: with shape N1 x M x N0
        :param mi_ub:
        :param kv_blk_idx:
        :return:
        """
        n1, m, n0 = Sij_ub_zN.shape
        # update_scale_ub: for update li and Oi(exp(mi_old - mi_new))
        update_scale_ub = self.tik_instance.Tensor(FP32, (m,), scope=UB, name="update_scale_ub")
        with self.tik_instance.if_scope(kv_blk_idx != 0):
            self.tik_instance.h_cast(update_scale_ub, mi_ub, "none")

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # TODO: 二分
            mn0_block_max = self.tik_instance.Tensor(FP16, (1, m, n0), name="mn0_block_max", scope=UB)
            self.cont_data_mv_1_bust(dst=mn0_block_max, src=Sij_ub_zN, burst=m)  # 用第一个(m, n0)始化，少一次duplicate和max
            with self.tik_instance.for_range(1, n1) as idx:
                self.tik_instance.h_max(mn0_block_max, mn0_block_max, Sij_ub_zN[idx, :, :])

            # reduce_max
            mn0_block_max = mn0_block_max.reshape((m, n0))
            mij_ub = self.tik_instance.Tensor(FP16, (m,), scope=UB, name="mij_ub")
            self.tik_instance.h_reduce_max(mij_ub, mn0_block_max, 1)
            # update mi
            with self.tik_instance.if_scope(kv_blk_idx == 0):
                self.cont_data_mv_1_bust(dst=mi_ub, src=mij_ub, burst=m // 16)
            with self.tik_instance.else_scope():
                self.tik_instance.h_max(mi_ub, mi_ub, mij_ub)

        with self.tik_instance.if_scope(kv_blk_idx != 0):
            # exp(mi_old - mi_new)
            mi_ub_fp32 = self.tik_instance.Tensor(FP32, (m,), scope=UB, name="mi_ub_fp32")
            self.tik_instance.h_cast(mi_ub_fp32, mi_ub, "none")
            self.tik_instance.h_sub(update_scale_ub, update_scale_ub, mi_ub_fp32)
            self.tik_instance.h_exp(update_scale_ub, update_scale_ub)

        return mi_ub, update_scale_ub

    def rowsum_compute(self, Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx):
        """Calculate row sum for softmax of flashattention2
        :param Pij_l1_zN: with shape N1 x M x N0
        :param li_ub:
        :param update_scale_ub:
        :param kv_blk_idx:
        :return:
        """
        n1, m, n0 = Pij_l1_zN.shape
        # cube impl rowsum of Pij
        lij_ub = self.tik_instance.Tensor(FP32, (m,), scope=UB, name="lij_ub")
        if n1 * n0 == self.Bc:
            lij_ub = self.tik_ops_utils.row_sum_cube_impl(Pij_l1_zN, self.ones_l1,
                                                          lij_ub, m, self.Bc, precision_type=FP32)
        else:
            lij_ub = self.tik_ops_utils.row_sum_cube_impl(Pij_l1_zN, self.last_ones_l1,
                                                          lij_ub, m, self.last_Bc, precision_type=FP32)

        # update li
        with self.tik_instance.if_scope(kv_blk_idx == 0):
            self.cont_data_mv_1_bust(dst=li_ub, src=lij_ub, burst=m // 8)
        with self.tik_instance.else_scope():
            self.tik_instance.h_mul(li_ub, li_ub, update_scale_ub)
            self.tik_instance.h_add(li_ub, li_ub, lij_ub)

        return li_ub

    def softmax_compute(self, Sij_ub_zN, li_ub, mi_ub, kv_blk_idx, drop_mask_gm_offset):
        """Calculate softmax
        :param Sij_ub_zN: with shape N1 x M x N0
        :param li_ub:
        :param mi_ub:
        :param kv_blk_idx:
        :param drop_mask_gm_offset:
        :return:
        """
        # rowmax
        mi_ub, update_scale_ub = self.rowmax_compute(Sij_ub_zN, mi_ub, kv_blk_idx)
        # Sij - mi
        n1, m, n0 = Sij_ub_zN.shape
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_mij_ub = self.tik_ops_utils.broadcast_vec_from_M_to_MN0(mi_ub)
            broadcast_mij_ub = broadcast_mij_ub.reshape((1, m, n0))
            for idx in range(n1):
                self.tik_instance.h_sub(Sij_ub_zN[idx, :, :], Sij_ub_zN[idx, :, :], broadcast_mij_ub)

        # exp, in place on the fp16 tile.
        #
        # The original path cast the whole score tile to a fresh fp32 UB buffer,
        # exponentiated there and cast back. That temporary, not the score tile
        # itself, was what capped the tiling: at Br=64/Bc=608 it is 152 KB and at
        # Bc=832 it is 208 KB, which is exactly where the build failed. Doing the
        # exp in place frees that buffer and lets Br double at the same footprint,
        # which halves the number of K/V sweeps.
        #
        # Precision cost: after the row-max subtraction the scores are <= 0, so
        # exp is in [0, 1]. fp16 holds that range well but only to ~1e-3 relative,
        # against ~5e-4 for the fp32 path. Verify at the real shape.
        self.tik_instance.h_exp(Sij_ub_zN, Sij_ub_zN)

        # rowsum
        Pij_l1_zN = self.tik_instance.Tensor(FP16, (n1, m, n0), scope=L1, name="Pij_l1_zN")
        self.cont_data_mv_1_bust(dst=Pij_l1_zN, src=Sij_ub_zN, burst=n1 * m * n0 // 16)
        li_ub = self.rowsum_compute(Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx)

        # dropout
        if self.has_drop_mask:
            self.do_dropout_mask(Sij_ub_zN, drop_mask_gm_offset, n1 * n0, n1 * n0, m, m)
            self.cont_data_mv_1_bust(dst=Pij_l1_zN, src=Sij_ub_zN, burst=n1 * m * n0 // 16)

        return Pij_l1_zN, li_ub, mi_ub, update_scale_ub

    def output_update(self, Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub, offset, n1_begin, n1_size, m):
        """Update output of flashattention2
        :param Oi_l1_zN: with shape N1 x M x N0
        :param Oij_ub_zN:
        :param broadcast_scale_ub:
        :param offset:
        :param n1_begin:
        :param n1_size:
        :param m:
        :return:
        """
        n1_end = n1_begin + n1_size
        Oi_ub_zN = self.tik_instance.Tensor(FP32, (n1_size, m, self.N0), scope=UB, name="Oi_ub_zN")
        # mte1: L1 -> UB
        self.cont_data_mv_1_bust(dst=Oi_ub_zN, src=Oi_l1_zN[offset], burst=n1_size * m * self.N0 // 8)
        # vector: mul and add
        for i in range(n1_size):
            self.tik_instance.h_mul(Oi_ub_zN[i, :, :], Oi_ub_zN[i, :, :], broadcast_scale_ub)
        self.tik_instance.h_add(Oi_ub_zN, Oi_ub_zN, Oij_ub_zN[n1_begin:n1_end, :, :])
        # mte3: UB -> L1
        self.cont_data_mv_1_bust(dst=Oi_l1_zN[offset], src=Oi_ub_zN, burst=n1_size * m * self.N0 // 8)

    def output_compute(self, Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, kv_blk_h):
        """Calculate output of flashattention2
        :param Oi_l1_zN: with shape N1 x M x N0
        :param Pij_l1_zN:
        :param Vj_l1_zN:
        :param update_scale_ub:
        :param q_blk_h:
        :param kv_blk_idx:
        :param kv_blk_h:
        :return:
        """
        m = Pij_l1_zN.shape[1]
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            Oij_ub_zN = self.tik_ops_utils.mmad_compute(Pij_l1_zN, Vj_l1_zN, q_blk_h, kv_blk_h, self.actual_d,
                                                        N1MN0_to_MN=False, precision_type=FP32)
            with self.tik_instance.if_scope(kv_blk_idx == 0):
                self.cont_data_mv_1_bust(dst=Oi_l1_zN, src=Oij_ub_zN, burst=m * self.d // 8)
            with self.tik_instance.else_scope():
                broadcast_scale_ub = self.tik_instance.Tensor(FP32, (1, m, self.N0), scope=UB,
                                                              name="broadcast_scale_ub")
                src_scalar = self.tik_instance.Scalar(FP32, name="src_scalar")
                with self.tik_instance.for_range(0, m) as idx:
                    src_scalar.set_as(update_scale_ub[idx])
                    self.tik_instance.h_duplicate(broadcast_scale_ub[0, idx, :], src_scalar)

                single_loop_size = 8 # prevent ub overflow
                loop_times, tail_num = divmod(self.N1, single_loop_size)
                with self.tik_instance.for_range(0, loop_times) as idx:
                    # double buffer
                    with self.tik_instance.for_range(0, 2, thread_num=2) as t_idx:
                        single_thread_size = single_loop_size // 2
                        offset = idx * single_loop_size * m * self.N0 + t_idx * single_thread_size * m * self.N0
                        n1_begin = idx * single_loop_size + t_idx * single_thread_size
                        self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub,
                                           offset, n1_begin, single_thread_size, m)
                if tail_num > 0:
                    offset = loop_times * single_loop_size * m * self.N0
                    thread_num = min(tail_num, 2)
                    # thread1
                    n1_begin = loop_times * single_loop_size
                    n1_size1 = tail_num // thread_num
                    self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub, offset, n1_begin, n1_size1, m)
                    # thread2
                    n1_size2 = tail_num - n1_size1
                    if n1_size2 > 0:
                        offset = offset + (tail_num // thread_num) * m * self.N0
                        n1_begin = loop_times * single_loop_size + tail_num // thread_num
                        self.output_update(Oi_l1_zN, Oij_ub_zN, broadcast_scale_ub, offset, n1_begin, n1_size2, m)
        return Oi_l1_zN

    def output_norm(self, Oi_l1_zN, broadcast_li_rec_ub, qo_gm_offset, o_l1_offset, n1_size, m):
        """Normalize output of flashattention2
        :param Oi_l1_zN: with shape N1 x M x N0
        :param broadcast_li_rec_ub:
        :param qo_gm_offset:
        :param o_l1_offset:
        :param n1_size:
        :param m:
        :return:
        """
        Oi_ub_zN = self.tik_instance.Tensor(FP32, (n1_size, m, self.N0), scope=UB, name="Oi_ub_zN")
        # mte1: L1 -> UB
        self.cont_data_mv_1_bust(dst=Oi_ub_zN, src=Oi_l1_zN[o_l1_offset], burst=n1_size * m * self.N0 // 8)
        # vector: mul and cast
        for i in range(n1_size):
            self.tik_instance.h_mul(Oi_ub_zN[i, :, :], Oi_ub_zN[i, :, :], broadcast_li_rec_ub)
        Oi_ub_zN_fp16 = self.tik_instance.Tensor(FP16, (n1_size, m, self.N0), scope=UB, name="Oi_ub_zN_fp16")
        self.tik_instance.h_cast(Oi_ub_zN_fp16, Oi_ub_zN, "none")
        # mte3: ub -> out
        self.unroll_data_move(dst=self.O_gm,
                              dst_offset=qo_gm_offset,
                              src=Oi_ub_zN_fp16,
                              src_offset=0,
                              nburst=n1_size,
                              burst=m * self.N0 // 16,
                              src_stride=0,
                              dst_stride=(self.Nq - m) * self.N0 // 16,
                              dst_offset_inc=self.Nq * self.N0,
                              src_offset_inc=m * self.N0
                              )

    def output_norm_and_write(self, Oi_l1_zN, li_ub, mi_ub, qo_gm_offset, l_gm_offset, q_blk_h):
        """Normalize output and write it back to gm
        :param Oi_l1_zN: with shape N1 x M x N0
        :param li_ub:
        :param mi_ub:
        :param qo_gm_offset:
        :param l_gm_offset:
        :param q_blk_h:
        :return:
        """
        m = Oi_l1_zN.shape[1]
        li_rec_ub = self.tik_ops_utils.calc_vec_rec(li_ub, m)
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            broadcast_li_rec_ub = self.tik_instance.Tensor(FP32, (1, m, self.N0), scope=UB, name="broadcast_li_rec_ub")
            src_scalar = self.tik_instance.Scalar(FP32, name="src_scalar")
            with self.tik_instance.for_range(0, m) as idx:
                src_scalar.set_as(li_rec_ub[idx])
                self.tik_instance.h_duplicate(broadcast_li_rec_ub[0, idx, :], src_scalar)
            single_loop_size = 12  # prevent ub overflow
            loop_times, tail_num = divmod(self.N1, single_loop_size)
            with self.tik_instance.for_range(0, loop_times) as idx:
                # double buffer
                with self.tik_instance.for_range(0, 2, thread_num=2) as t_idx:
                    single_thread_size = single_loop_size // 2
                    cur_qo_gm_offset = qo_gm_offset + \
                                       (idx * single_loop_size + t_idx * single_thread_size) * self.Nq * self.N0
                    o_l1_offset = (idx * single_loop_size + t_idx * single_thread_size) * m * self.N0
                    self.output_norm(Oi_l1_zN, broadcast_li_rec_ub,
                                     cur_qo_gm_offset, o_l1_offset, single_thread_size, m)
            if tail_num > 0:
                thread_num = min(tail_num, 2)
                # thread1
                cur_qo_gm_offset1 = qo_gm_offset + loop_times * single_loop_size * self.Nq * self.N0
                o_l1_offset1 = loop_times * single_loop_size * m * self.N0
                n1_size1 = tail_num // thread_num
                self.output_norm(Oi_l1_zN, broadcast_li_rec_ub, cur_qo_gm_offset1, o_l1_offset1, n1_size1, m)
                # thread2
                n1_size2 = tail_num - n1_size1
                if n1_size2 > 0:
                    cur_qo_gm_offset2 = cur_qo_gm_offset1 + n1_size1 * self.Nq * self.N0
                    o_l1_offset2 = o_l1_offset1 + n1_size1 * m * self.N0
                    self.output_norm(Oi_l1_zN, broadcast_li_rec_ub, cur_qo_gm_offset2, o_l1_offset2, n1_size2, m)

        # compute li
        mi_ub_fp32 = self.tik_instance.Tensor(FP32, (m,), scope=UB, name="mi_ub_fp32")
        self.tik_instance.h_cast(mi_ub_fp32, mi_ub, "none")
        self.tik_instance.h_ln(li_ub, li_ub)
        self.tik_instance.h_add(li_ub, li_ub, mi_ub_fp32)
        self.tik_ops_utils.move_vector_from_ub_to_gm(self.l_gm, li_ub, l_gm_offset, q_blk_h)

    def compute_in_each_kv_block(self, Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start,
                                 batch_idx, q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h, workspace=None):
        """The forward computation in each inner loop"""
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_h)
        kv_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(kv_blk_h)
        kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc,
                                          kv_blk_idx)

        # load Vj (kv_blk_idx_th block of V_gm), then reorder for Pij*Vj
        Vj_l1_zN = self.tik_instance.Tensor(FP16, (kv_blk_h_aligned, self.d), name="Vj_l1_zN", scope=L1)
        self.unroll_data_move(dst=Vj_l1_zN,
                              dst_offset=0,
                              src=self.V_gm,
                              src_offset=kv_gm_offset,
                              nburst=self.N1,
                              burst=kv_blk_h_aligned * self.N0 // 16,
                              src_stride=(self.N - kv_blk_h_aligned) * self.N0 // 16,
                              dst_stride=0,
                              dst_offset_inc=kv_blk_h_aligned * self.N0,
                              src_offset_inc=self.N * self.N0
                              )

        if workspace is None:
            # load Kj (kv_blk_idx_th block of K_gm)
            KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, kv_blk_h_aligned, self.N0),
                                                       name="KjT_l1_K1MK0_ed", scope=L1)
            self.unroll_data_move(dst=KjT_l1_zN,
                                  dst_offset=0,
                                  src=self.K_gm,
                                  src_offset=kv_gm_offset,
                                  nburst=self.N1,
                                  burst=kv_blk_h_aligned * self.N0 // 16,
                                  src_stride=(self.N - kv_blk_h_aligned) * self.N0 // 16,
                                  dst_stride=0,
                                  dst_offset_inc=kv_blk_h_aligned * self.N0,
                                  src_offset_inc=self.N * self.N0
                                  )
            # Qi*Kj.T
            Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN,
                                                       q_blk_h, self.actual_d, kv_blk_h, N1MN0_to_MN=False)
        else:
            Sij_ub_zN = workspace

        # Sij <- Sij + attn_mask
        if self.has_attn_mask:
            if self.only_causal:
                with self.tik_instance.if_scope(q_blk_idx == kv_blk_idx): # main diagonal
                    self.do_att_mask(Sij_ub_zN, 0, q_blk_h, kv_blk_h,
                                     q_blk_h_aligned, kv_blk_h_aligned, self.only_causal)
            else:
                attn_mask_gm_offset = self.get_attn_mask_gm_offset(batch_start, batch_idx, self.Nq, self.N,
                                                               self.Br, q_blk_idx, self.Bc, kv_blk_idx)
                self.do_att_mask(Sij_ub_zN, attn_mask_gm_offset, q_blk_h, kv_blk_h,
                                 q_blk_h_aligned, kv_blk_h_aligned)
        # Pij = softmax(Sij), Pij <- dropout(Pij)
        drop_mask_gm_offset = None
        if self.has_drop_mask:
            drop_mask_gm_offset = self.get_drop_mask_gm_offset(batch_start, batch_idx, self.Nq,
                                                                  self.N, self.Br, q_blk_idx, self.Bc, kv_blk_idx)
        Pij_l1_zN, li_ub, mi_ub, update_scale_ub = self.softmax_compute(Sij_ub_zN, li_ub, mi_ub,
                                                                     kv_blk_idx, drop_mask_gm_offset)

        Oi_l1_zN = self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, kv_blk_h)

        # do Sij = matmul(Qi, Kj.T) for next inner loop
        with self.tik_instance.if_scope(kv_blk_idx < self.Tc - 2):
            kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc,
                                              kv_blk_idx + 1)
            KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, self.Bc, self.N0),
                                                       name="KjT_l1_zN", scope=L1)
            self.unroll_data_move(dst=KjT_l1_zN,
                                  dst_offset=0,
                                  src=self.K_gm,
                                  src_offset=kv_gm_offset,
                                  nburst=self.N1,
                                  burst=self.Bc * self.N0 // 16,
                                  src_stride=(self.N - self.Bc) * self.N0 // 16,
                                  dst_stride=0,
                                  dst_offset_inc=self.Bc * self.N0,
                                  src_offset_inc=self.N * self.N0
            )
            Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN, KjT_l1_zN, q_blk_h, self.actual_d, self.Bc,
                                                          N1MN0_to_MN=False, workspace=Sij_ub_zN)

        return Oi_l1_zN, li_ub, mi_ub

    def compute_in_each_q_block(self, batch_start, batch_idx, q_blk_idx, q_blk_h):
        """The forward computation in each outer loop"""
        q_blk_h_aligned = self.tik_ops_utils.up_align_to_K0(q_blk_h)

        # initialize Oi_l1, li_ub, mi_ub
        Oi_l1_zN = self.tik_instance.Tensor(FP32, (self.d // self.N0, q_blk_h_aligned, self.N0),
                                         scope=L1, name="Oi_l1_zN")
        li_ub = self.tik_instance.Tensor(FP32, (q_blk_h_aligned,), scope=UB, name="li_ub")
        mi_ub = self.tik_instance.Tensor(FP16, (q_blk_h_aligned,), scope=UB, name="mi_ub")

        # load Qi (q_blk_idx_th block of Q_gm)
        qo_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.Nq, self.d, self.Br, q_blk_idx)
        Qi_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, q_blk_h_aligned, self.N0),
                                                  scope=L1, name="Qi_l1_zN")
        self.unroll_data_move(dst=Qi_l1_zN,
                              dst_offset=0,
                              src=self.Q_gm,
                              src_offset=qo_gm_offset,
                              nburst=self.N1,
                              burst=q_blk_h_aligned * self.N0 // 16,
                              src_stride=(self.Nq - q_blk_h_aligned) * self.N0 // 16,
                              dst_stride=0,
                              dst_offset_inc=q_blk_h_aligned * self.N0,
                              src_offset_inc=self.Nq * self.N0
                              )

        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # load K0 (0_th block of K_gm)
            kv_gm_offset = self.get_gm_offset(batch_start, batch_idx, self.N, self.d, self.Bc, 0)
            # zN等价于转置后的K1NK0
            KjT_l1_zN = self.tik_instance.Tensor(FP16, (self.d // self.N0, self.Bc, self.N0),
                                                       name="KjT_l1_zN", scope=L1)
            self.unroll_data_move(dst=KjT_l1_zN,
                                  dst_offset=0,
                                  src=self.K_gm,
                                  src_offset=kv_gm_offset,
                                  nburst=self.N1,
                                  burst=self.Bc * self.N0 // 16,
                                  src_stride=(self.N - self.Bc) * self.N0 // 16,
                                  dst_stride=0,
                                  dst_offset_inc=self.Bc * self.N0,
                                  src_offset_inc=self.N * self.N0
                                  )
            # do Sij = matmul(Qi, Kj.T) for first inner loop
            Sij_ub_zN = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_ub_zN")
            Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN,
                                                          KjT_l1_zN,
                                                          q_blk_h, self.actual_d, self.Bc,
                                                          N1MN0_to_MN=False, workspace=Sij_ub_zN)
            with self.tik_instance.for_range(0, self.Tc - 1, name="kv_blk_idx") as kv_blk_idx:
                with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,
                                                        q_blk_idx <= kv_blk_idx + self.prev_block_num)):
                    Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub,
                                                                           batch_start, batch_idx, q_blk_idx, kv_blk_idx,
                                                                           q_blk_h, self.Bc, workspace=Sij_ub_zN)
        # last kv blk
        with self.tik_instance.if_scope(tik.all(self.Tc - 1 - self.next_block_num <= q_blk_idx,
                                                q_blk_idx <= self.Tc - 1 + self.prev_block_num)):
            Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start,
                                                                   batch_idx, q_blk_idx, self.Tc - 1, q_blk_h,
                                                                   self.last_Bc)
        # normalize output and write it to gm
        l_gm_offset = self.get_l_gm_offset(batch_start, batch_idx, self.Nq, self.Br, q_blk_idx)
        self.output_norm_and_write(Oi_l1_zN, li_ub, mi_ub, qo_gm_offset, l_gm_offset, q_blk_h)

    def compute_one_core(self, batch_start_sc, batch_num_sc, core_idx_to_tr_info, core_idx):
        """The computation of FlashAttention forward on each core"""
        with self.tik_instance.for_range(0, batch_num_sc, name="batch_index") as batch_idx:
            tr_start_s = self.tik_instance.Scalar("int32",
                                                  init_value=core_idx_to_tr_info[
                                                      core_idx, batch_start_sc + batch_idx, 0],
                                                  name="tr_start_s")
            tr_end_s = self.tik_instance.Scalar("int32",
                                                init_value=core_idx_to_tr_info[core_idx, batch_start_sc + batch_idx, 1],
                                                name="tr_end_s")
            with self.tik_instance.for_range(tr_start_s, tr_end_s - 1, name="q_blk_idx") as q_blk_idx:
                self.compute_in_each_q_block(batch_start_sc, batch_idx, q_blk_idx, self.Br)
            with self.tik_instance.if_scope(tr_end_s - 1 != self.Tr - 1):
                self.compute_in_each_q_block(batch_start_sc, batch_idx, tr_end_s - 1, self.Br)
            with self.tik_instance.else_scope():
                self.compute_in_each_q_block(batch_start_sc, batch_idx, self.Tr - 1, self.last_Br)

    def collect_inputs(self):
        """collect all input gm tensors into input_gm_list,
        the input list should keep order with the para order in Primitive and init
        """
        input_gm_list = [self.Q_gm, self.K_gm, self.V_gm]
        if self.has_attn_mask:
            input_gm_list.append(self.att_mask_gm)
        if self.has_drop_mask:
            input_gm_list.append(self.drop_mask_gm)
        if self.has_alibi_mask:
            input_gm_list.append(self.alibi_mask_gm)

        return input_gm_list

    def collect_outputs(self):
        """collect all output gm tensors into output_gm_list,
        the output list should keep order with the para order in Primitive and init
        """
        output_gm_list = [self.O_gm, self.l_gm]
        return output_gm_list


def flash_attention(query, key, value, attn_mask, dropout_mask, alibi_mask, output, rowsum,
                    prev_block_num=65536, next_block_num=65536, tiling_stgy_name='sparse',
                    kernel_name="flash_attention", disable_debug=True):
    """
    algorithm: flash_attention_backward

    Parameters
    ----------
    query : dict. shape and dtype of input, only support float16
    key : dict. shape and dtype of input, only support float16
    value: dict. shape and dtype of input, only support float16
    attn_mask: dict. shape and dtype of input, only support float16
    dropout_mask: dict. shape and dtype of input, only support float16
    alibi_mask: dict. shape and dtype of input, only support float16
    output: dict. shape and dtype of output, only support float16
    rowsum: dict. shape and dtype of output, only support float16
    prev_block_num: int. an attribute used to define sparse attention
    next_block_num: int. an attribute used to define sparse attention
    tiling_stgy_name: str. an attribute used to choose the tiling strategy
    kernel_name: str. cce kernel name, default value is real_div
    disable_debug: bool. whether disable debug

    Returns
    -------
    tik_instance
    """
    fa = FlashAttentionFwd(q=query, k=key, v=value, attn_mask=attn_mask,
                           dropout_mask=dropout_mask, alibi_mask=alibi_mask, kernel_name=kernel_name,
                           tiling_stgy=TilingStrategy.from_strategy_name(tiling_stgy_name),
                           prev_block_num=prev_block_num, next_block_num=next_block_num,
                           disable_debug=disable_debug)
    fa.compute_process()
    return fa.tik_instance

