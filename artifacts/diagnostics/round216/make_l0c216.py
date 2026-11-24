"""Build the R216 explicit L0C-workspace prototype from pinned original sources.

Every requested replacement is asserted unique. The generated candidate is not used by any
production path; it is compiled only by the H3_FA215_PKG isolated overlay.
"""
from pathlib import Path
import ast

ROOT = Path('/Users/lvyufeng/Projects/h3-910a')
R215 = ROOT / 'artifacts/diagnostics/round215'
R216 = ROOT / 'artifacts/diagnostics/round216'


def replace_once(text, old, new):
    assert text.count(old) == 1, f'Expected one match, got {text.count(old)} for {old[:100]!r}'
    return text.replace(old, new, 1)


u = (R215 / 'tik_ops_utils_original.py').read_text()
u = replace_once(u, 'from tbe import tik\n',
                 'from tbe import tik\nfrom tbe.tik.api.cube.matmul import MatMulImpl\n'
                 'from tbe.tik.api.cube.tik_cube_api import MatmulApiCls\n')
u = replace_once(u,
    '    def row_sum_cube_impl(self, matrix_l1_K1MK0_ed, right_all_one_matrix_l1, rowsum_ub, m, k, precision_type):',
    '    def row_sum_cube_impl(self, matrix_l1_K1MK0_ed, right_all_one_matrix_l1, rowsum_ub, m, k, precision_type, '
    'l0c_work=None):')
u = replace_once(u,
    'self.matmul_compute(matrix_l1_K1MK0_ed, right_all_one_matrix_l1, m, k, 16,\n'
    '                                                   N1MN0_to_MN=False, precision_type=precision_type)',
    'self.matmul_compute(matrix_l1_K1MK0_ed, right_all_one_matrix_l1, m, k, 16,\n'
    '                                                   N1MN0_to_MN=False, precision_type=precision_type, '
    'l0c_work=l0c_work)')
u = replace_once(u,
    '    def matmul_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=True, precision_type=FP16, workspace=None):',
    '    def matmul_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=True, precision_type=FP16, workspace=None, '
    'l0c_work=None):')
u = replace_once(u,
    '    def mmad_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=False, precision_type=FP16):',
    '    def mmad_compute(self, A_l1, B_l1, m, k, n, N1MN0_to_MN=False, precision_type=FP16, l0c_work=None):')
old_c = '''            C_l0c = self.tik_instance.Tensor(FP32, (N // 16, M, 16), scope=L0C, name="C_l0c")  # n1mn0 (n0=16)'''
new_c = '''            C_l0c = l0c_work[:N // 16, :M, :16]'''
assert old_c in u
u = u.replace(old_c, new_c)
old_mm = '''        with self.tik_instance.new_stmt_scope(disable_sync=False):
            # matmul
            if K > 16:'''
new_mm = '''        C_l0c = l0c_work[:N // 16, :M, :16]
        with self.tik_instance.new_stmt_scope(disable_sync=False):
            if K > 16:'''
u = replace_once(u, old_mm, new_mm)
old_tm = '''            C_l0c = self.tik_instance.Tensor(
                FP32, (N // 16, M, 16), scope=L0C, name="C_l0c"
            )  # n1mn0 (n0=16)
            self.tik_instance.matmul(C_l0c, A_l1, B_l1, m, k, n)'''
new_tm = '''            C_l0c = l0c_work[:N // 16, :M, :16]
            matmul_impl = MatMulImpl(True, C_l0c, self.tik_instance)
            matmul_impl.execute(MatmulApiCls(A_l1, B_l1, m, k, n, None))'''
u = replace_once(u, old_tm, new_tm)
(R216 / 'tik_ops_utils_l0c.py').write_text(u)

f = (R215 / 'flash_attention_fwd_original.py').read_text()
f = replace_once(f,
    'from flash_attention.ops.flash_attention.constants import UB\n',
    'from flash_attention.ops.flash_attention.constants import UB\n'
    'from flash_attention.ops.flash_attention.constants import L0C\n')
f = replace_once(f,
    '    def rowsum_compute(self, Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx):',
    '    def rowsum_compute(self, Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx, l0c_work):')
f = replace_once(f, 'self.ones_l1,', 'self.ones_l1, l0c_work=l0c_work,')
f = replace_once(f, 'self.last_ones_l1,', 'self.last_ones_l1, l0c_work=l0c_work,')
f = replace_once(f,
    '    def softmax_compute(self, Sij_ub_zN, li_ub, mi_ub, kv_blk_idx, drop_mask_gm_offset):',
    '    def softmax_compute(self, Sij_ub_zN, li_ub, mi_ub, kv_blk_idx, drop_mask_gm_offset, l0c_work):')
f = replace_once(f,
    'self.rowsum_compute(Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx)',
    'self.rowsum_compute(Pij_l1_zN, li_ub, update_scale_ub, kv_blk_idx, l0c_work)')
f = replace_once(f,
    '    def output_compute(self, Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, kv_blk_h):',
    '    def output_compute(self, Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, '
    'kv_blk_h, l0c_work):')
f = replace_once(f, 'N1MN0_to_MN=False, precision_type=FP32)',
                 'N1MN0_to_MN=False, precision_type=FP32, l0c_work=l0c_work)')
f = replace_once(f,
    'q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h, workspace=None):',
    'q_blk_idx, kv_blk_idx, q_blk_h, kv_blk_h, workspace=None, l0c_work=None):')
f = replace_once(f, '                                                 N1MN0_to_MN=False)',
                     '                                                 N1MN0_to_MN=False, l0c_work=l0c_work)')
f = replace_once(f, 'kv_blk_idx, drop_mask_gm_offset)', 'kv_blk_idx, drop_mask_gm_offset, l0c_work)')
f = replace_once(f,
    'N1MN0_to_MN=False, workspace=Sij_ub_zN)',
    'N1MN0_to_MN=False, workspace=Sij_ub_zN, l0c_work=l0c_work)')
f = replace_once(f,
    'self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, kv_blk_h)',
    'self.output_compute(Oi_l1_zN, Pij_l1_zN, Vj_l1_zN, update_scale_ub, q_blk_h, kv_blk_idx, '
    'kv_blk_h, l0c_work)')
f = replace_once(f,
    '                                            q_blk_h, self.Bc, workspace=Sij_ub_zN)',
    '                                            q_blk_h, self.Bc, workspace=Sij_ub_zN, l0c_work=l0c_work)')
f = replace_once(f,
    '        # load Qi (q_blk_idx_th block of Q_gm)',
    '        l0c_work = self.tik_instance.Tensor(FP32, (self.Bc // 16, q_blk_h_aligned, 16), '
    'scope=L0C, name="l0c_work")\n'
    '        # load Qi (q_blk_idx_th block of Q_gm)')
f = replace_once(f,
    '                                                           N1MN0_to_MN=False, workspace=Sij_ub_zN)',
    '                                                           N1MN0_to_MN=False, workspace=Sij_ub_zN, '
    'l0c_work=l0c_work)')
(R216 / 'flash_attention_fwd_l0c.py').write_text(f)
for name in ['tik_ops_utils_l0c.py', 'flash_attention_fwd_l0c.py']:
    path = R216 / name
    ast.parse(path.read_text(), filename=str(path))
    print('generated valid source:', path, path.stat().st_size)
