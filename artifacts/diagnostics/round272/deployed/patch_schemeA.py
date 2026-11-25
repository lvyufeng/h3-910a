#!/usr/bin/env python3
"""R272: apply scheme-A double-buffering patch to the PRIVATE FA copy.

Edits (see round271/PATCH_SPEC.md):
  1. compute_in_each_kv_block gains `workspace_next=None`.
  2. the "prefetch K_{j+1} + S_{j+1}" block is MOVED to before softmax_j, writing workspace_next.
  3. compute_in_each_q_block allocates Sij_a/Sij_b and unrolls the KV loop by 2.
Strict assertions: every anchor must match exactly once; the file is written only if all pass.
No source outside the private copy is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/tmp/fa_pkg_r266/flash_attention/ops/flash_attention/flash_attention_fwd.py")
src = P.read_text()
orig = src
report = []

# ---- patch 1: signature -------------------------------------------------------
sig_old = ("    def compute_in_each_kv_block(self, Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start,\n"
           "                                 batch_idx, q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h, workspace=None):")
sig_new = ("    def compute_in_each_kv_block(self, Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start,\n"
           "                                 batch_idx, q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h,\n"
           "                                 workspace=None, workspace_next=None):")
assert src.count(sig_old) == 1, "patch1 anchor count=%d" % src.count(sig_old)
src = src.replace(sig_old, sig_new)
report.append("patch1 signature ok")

# ---- patch 2: move prefetch+S before softmax, writing workspace_next ----------
lines = src.split("\n")
start = next(i for i, l in enumerate(lines)
             if "# do Sij = matmul(Qi, Kj.T) for next inner loop" in l)
end = next(i for i in range(start, len(lines))
           if "N1MN0_to_MN=False, workspace=Sij_ub_zN)" in lines[i])
block = lines[start:end + 1]
assert any("if_scope(kv_blk_idx < self.Tc - 2)" in l for l in block), "patch2 no if_scope in block"
assert any("unroll_data_move" in l for l in block), "patch2 no data move"
block = [l.replace("workspace=Sij_ub_zN)",
                   "workspace=(workspace_next if workspace_next is not None else Sij_ub_zN))")
         for l in block]
del lines[start:end + 1]
soft = next(i for i, l in enumerate(lines)
            if "Pij_l1_zN, li_ub, mi_ub, update_scale_ub = self.softmax_compute" in l)
lines[soft:soft] = block
src = "\n".join(lines)
report.append("patch2 moved %d lines before softmax (line %d)" % (len(block), soft))

# ---- patch 3: dual S buffers + unroll by 2 -----------------------------------
q_old = (
    '            Sij_ub_zN = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_ub_zN")\n'
    '            Sij_ub_zN = self.tik_ops_utils.matmul_compute(Qi_l1_zN,\n'
    '                                                          KjT_l1_zN,\n'
    '                                                          q_blk_h, self.actual_d, self.Bc,\n'
    '                                                          N1MN0_to_MN=False, workspace=Sij_ub_zN)\n'
    '            with self.tik_instance.for_range(0, self.Tc - 1, name="kv_blk_idx") as kv_blk_idx:\n'
    '                with self.tik_instance.if_scope(tik.all(kv_blk_idx - self.next_block_num <= q_blk_idx,\n'
    '                                                        q_blk_idx <= kv_blk_idx + self.prev_block_num)):\n'
    '                    Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub,\n'
    '                                                                           batch_start, batch_idx, q_blk_idx, kv_blk_idx,\n'
    '                                                                           q_blk_h, self.Bc, workspace=Sij_ub_zN)')
q_new = (
    '            Sij_a = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_a")\n'
    '            Sij_b = self.tik_instance.Tensor(FP16, (self.Bc // 16, q_blk_h_aligned, 16), scope=UB, name="Sij_b")\n'
    '            Sij_a = self.tik_ops_utils.matmul_compute(Qi_l1_zN,\n'
    '                                                      KjT_l1_zN,\n'
    '                                                      q_blk_h, self.actual_d, self.Bc,\n'
    '                                                      N1MN0_to_MN=False, workspace=Sij_a)\n'
    '            n_kv_blk = self.Tc - 1\n'
    '            with self.tik_instance.for_range(0, (n_kv_blk + 1) // 2, name="kv_pair") as kv_pair:\n'
    '                j0 = kv_pair * 2\n'
    '                with self.tik_instance.if_scope(j0 < n_kv_blk):\n'
    '                    with self.tik_instance.if_scope(tik.all(j0 - self.next_block_num <= q_blk_idx,\n'
    '                                                            q_blk_idx <= j0 + self.prev_block_num)):\n'
    '                        Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(\n'
    '                            Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start, batch_idx, q_blk_idx,\n'
    '                            j0, q_blk_h, self.Bc, workspace=Sij_a, workspace_next=Sij_b)\n'
    '                with self.tik_instance.if_scope(j0 + 1 < n_kv_blk):\n'
    '                    with self.tik_instance.if_scope(tik.all(j0 + 1 - self.next_block_num <= q_blk_idx,\n'
    '                                                            q_blk_idx <= j0 + 1 + self.prev_block_num)):\n'
    '                        Oi_l1_zN, li_ub, mi_ub = self.compute_in_each_kv_block(\n'
    '                            Qi_l1_zN, Oi_l1_zN, li_ub, mi_ub, batch_start, batch_idx, q_blk_idx,\n'
    '                            j0 + 1, q_blk_h, self.Bc, workspace=Sij_b, workspace_next=Sij_a)')
assert src.count(q_old) == 1, "patch3 anchor count=%d" % src.count(q_old)
src = src.replace(q_old, q_new)
report.append("patch3 dual buffers + unroll ok")

assert src != orig
backup = P.with_suffix(".py.orig")
if not backup.exists():
    backup.write_text(orig)
P.write_text(src)
print("\n".join(report))
print("patched", P, "backup", backup.name)
