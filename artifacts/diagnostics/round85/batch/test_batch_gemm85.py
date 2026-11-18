#!/usr/bin/env python3
"""Stdlib-only tests for the Round85 batched per-head FP32-output GEMM builder.

No torch, no CANN, no NPU and no network are used: import of ``batch_gemm85``
itself needs none of them. These tests verify the *arithmetic contracts* the
runtime path depends on - tile plan, per-head GM offsets, DMA strides, host
layout permutations, task flattening, no-head-leak and source invariants. They
prove nothing about numerics or speed; those require the root's actual run.
"""

import ast
import hashlib
import importlib
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
R83 = REPO / "artifacts/diagnostics/round83/blocked_dynamic_steps83.py"

sys.path.insert(0, str(HERE))
m = importlib.import_module("batch_gemm85")
sys.path.pop(0)

SOURCE = (HERE / "batch_gemm85.py").read_text()


# ---------------------------------------------------------------------------
# Pure-python reference implementations of the host permutations
# ---------------------------------------------------------------------------
# These replicate pack_a / pack_w / unpack_c index arithmetic WITHOUT torch, so
# the layout can be verified on a machine with no torch installed. Each returns
# a flat list in the packed GM order.


def ref_pack_a(a, h, mm, k, m_pad):
    """(H,M,K) nested list -> flat (H,K1,Mp,16) order, zero padded M rows."""
    k1 = k // 16
    out = [0.0] * (h * k1 * m_pad * 16)
    for hh in range(h):
        for mi in range(mm):
            for ki in range(k):
                val = a[hh][mi][ki]
                k1i, k0 = divmod(ki, 16)
                idx = (((hh * k1 + k1i) * m_pad + mi) * 16 + k0)
                out[idx] = val
    return out


def ref_pack_w(w, h, n, k):
    """(H,N,K) nested list -> flat (H,K1,N,16) order, values untouched."""
    k1 = k // 16
    out = [0.0] * (h * k1 * n * 16)
    for hh in range(h):
        for ni in range(n):
            for ki in range(k):
                val = w[hh][ni][ki]
                k1i, k0 = divmod(ki, 16)
                idx = (((hh * k1 + k1i) * n + ni) * 16 + k0)
                out[idx] = val
    return out


def ref_unpack_c(c_flat, h, n1, m_pad, m_out):
    """Flat (H,N1,Mp,16) -> (H,M,N) nested, cropping padded rows."""
    n = n1 * 16
    out = [[[0.0] * n for _ in range(m_out)] for _ in range(h)]
    for hh in range(h):
        for n1i in range(n1):
            for mi in range(m_out):
                for n0 in range(16):
                    idx = (((hh * n1 + n1i) * m_pad + mi) * 16 + n0)
                    out[hh][mi][n1i * 16 + n0] = c_flat[idx]
    return out


class PlanTilingTests(unittest.TestCase):
    def test_score_geometry_plan(self):
        p = m.plan_tiling(8, 512, 128, 2048)
        self.assertEqual(p["logical"], {"batch": 8, "m": 512, "k": 128, "n": 2048})
        self.assertEqual(p["padded"]["m_pad"], 512)
        self.assertEqual(p["padded"]["m_pad_rows"], 0)
        self.assertEqual(p["padded"]["k1"], 8)
        self.assertEqual(p["padded"]["n1"], 128)
        self.assertEqual(p["grid"]["m_tiles"], 4)
        self.assertEqual(p["grid"]["n_tiles"], 16)
        self.assertEqual(p["grid"]["per_head_tasks"], 64)
        self.assertEqual(p["grid"]["tasks"], 512)
        self.assertEqual(p["grid"]["cores"], 30)
        self.assertEqual(p["grid"]["steps"], math.ceil(512 / 30))
        self.assertEqual(p["accumulation"]["k_tiles"], 1)

    def test_pv_geometry_plan(self):
        p = m.plan_tiling(8, 512, 2048, 128)
        self.assertEqual(p["padded"]["k1"], 128)
        self.assertEqual(p["padded"]["n1"], 8)
        self.assertEqual(p["grid"]["m_tiles"], 4)
        self.assertEqual(p["grid"]["n_tiles"], 1)
        self.assertEqual(p["grid"]["per_head_tasks"], 4)
        self.assertEqual(p["grid"]["tasks"], 32)
        self.assertEqual(p["grid"]["cores"], 30)
        self.assertEqual(p["grid"]["steps"], 2)
        self.assertEqual(p["accumulation"]["k_tiles"], 16)

    def test_last_chunk_507_pads_m_to_512(self):
        for shape in ((8, 507, 128, 2048), (8, 507, 2048, 128)):
            p = m.plan_tiling(*shape)
            self.assertEqual(p["padded"]["m_pad"], 512)
            self.assertEqual(p["padded"]["m_pad_rows"], 5)
            self.assertEqual(p["grid"]["tasks"], p["logical"]["batch"] * p["grid"]["per_head_tasks"])

    def test_smoke_geometry_exercises_two_k_blocks_and_two_heads(self):
        h, mm, k, n = m.SUGGESTED_SMOKE_GEOMETRY
        p = m.plan_tiling(h, mm, k, n)
        self.assertEqual(p["accumulation"]["k_tiles"], 2)
        self.assertEqual(p["logical"]["batch"], 2)
        self.assertEqual(p["grid"]["cores"], 2)

    def test_cores_never_exceed_thirty_and_cover_all_tasks(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            self.assertLessEqual(p["grid"]["cores"], 30)
            self.assertGreaterEqual(p["grid"]["cores"] * p["grid"]["steps"], p["grid"]["tasks"])

    def test_invalid_shapes_fail_closed(self):
        for bad in ((0, 512, 128, 2048), (8, 512, 100, 2048), (8, 512, 128, 200),
                    (8, 512, 128, 2048, 0), (8, 512, 128, 2048, 128, 128, 100)):
            with self.assertRaises(m.BatchGemmError):
                m.plan_tiling(*bad)
        with self.assertRaises(m.BatchGemmError):
            m.plan_tiling(8, 512.0, 128, 2048)

    def test_gm_bytes_match_element_counts(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["pv"])
        self.assertEqual(p["gm_bytes"]["a"], p["gm_elems"]["a"] * 2)
        self.assertEqual(p["gm_bytes"]["b"], p["gm_elems"]["b"] * 2)
        self.assertEqual(p["gm_bytes"]["c"], p["gm_elems"]["c"] * 4)

    def test_head_stride_matches_per_head_shapes(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["score"])
        k1, m_pad, n, n1 = (p["padded"]["k1"], p["padded"]["m_pad"],
                            p["logical"]["n"], p["padded"]["n1"])
        self.assertEqual(p["head_stride_elems"]["a"], k1 * m_pad * 16)
        self.assertEqual(p["head_stride_elems"]["b"], k1 * n * 16)
        self.assertEqual(p["head_stride_elems"]["c"], n1 * m_pad * 16)
        self.assertEqual(p["gm_elems"]["a"], p["logical"]["batch"] * p["head_stride_elems"]["a"])

    def test_budget_is_within_l1_l0c_ub(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            m._check_budget(p)  # must not raise
            self.assertLessEqual(p["buffers"]["l1_a_bytes"] + p["buffers"]["l1_b_bytes"],
                                 m.PINNED_L1_SIZE)
            self.assertLessEqual(p["buffers"]["l0c_bytes"], m.PINNED_L0C_SIZE)


class TaskFlattenTests(unittest.TestCase):
    def test_flatten_is_bijective_over_all_heads_and_tiles(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["score"])
        batch, nt = p["logical"]["batch"], p["grid"]["n_tiles"]
        per_head = p["grid"]["per_head_tasks"]
        seen = set()
        for task in range(batch * per_head):
            head, mt, ntile = m.flatten_task(task, batch, nt, per_head)
            self.assertLess(head, batch)
            self.assertLess(mt, p["grid"]["m_tiles"])
            self.assertLess(ntile, nt)
            self.assertNotIn((head, mt, ntile), seen)
            seen.add((head, mt, ntile))
        self.assertEqual(len(seen), batch * p["grid"]["m_tiles"] * nt)

    def test_flatten_never_skips_a_head(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["pv"])
        batch, nt, per_head = p["logical"]["batch"], p["grid"]["n_tiles"], p["grid"]["per_head_tasks"]
        heads = [m.flatten_task(t, batch, nt, per_head)[0] for t in range(batch * per_head)]
        self.assertEqual(set(heads), set(range(batch)))
        for h in range(batch):
            self.assertEqual(heads.count(h), per_head)

    def test_out_of_range_tasks_are_rejected(self):
        for bad in (-1, 512, 513):
            with self.assertRaises(m.BatchGemmError):
                m.flatten_task(bad, 8, 16, 64)
        with self.assertRaises(m.BatchGemmError):
            m.flatten_task(1.0, 8, 16, 64)


class HeadOffsetTests(unittest.TestCase):
    def test_offsets_include_head_stride_for_a_b_and_c(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["score"])
        hs = p["head_stride_elems"]
        z = m.gm_offsets(p, 0, 0, 0, 0)
        one = m.gm_offsets(p, 1, 0, 0, 0)
        self.assertEqual(one["a"] - z["a"], hs["a"])
        self.assertEqual(one["b"] - z["b"], hs["b"])
        self.assertEqual(one["c"] - z["c"], hs["c"])

    def test_every_head_stays_inside_its_own_gm_region(self):
        """No head may read/write outside its own slice - this is the head leak check."""
        p = m.plan_tiling(*m.TARGET_SHAPES["score"])
        hs, pd = p["head_stride_elems"], p["padded"]
        c_elems = (p["tiles"]["nb"] // 16) * p["tiles"]["mb"] * 16
        for head in range(p["logical"]["batch"]):
            lo = {k: head * hs[k] for k in hs}
            hi = {k: (head + 1) * hs[k] for k in hs}
            for mt in range(p["grid"]["m_tiles"]):
                for nt in range(p["grid"]["n_tiles"]):
                    for kt in range(p["accumulation"]["k_tiles"]):
                        off = m.gm_offsets(p, head, mt, nt, kt)
                        # A tile: kb1*mb*16 elements starting at off['a']
                        a_span = (p["tiles"]["kb"] // 16) * p["tiles"]["mb"] * 16
                        self.assertGreaterEqual(off["a"], lo["a"])
                        self.assertLessEqual(off["a"] + a_span, hi["a"])
                        self.assertGreaterEqual(off["b"], lo["b"])
                        self.assertLessEqual(off["b"] + a_span, hi["b"])
                        self.assertGreaterEqual(off["c"], lo["c"])
                        self.assertLessEqual(off["c"] + c_elems, hi["c"])

    def test_all_heads_covered_and_total_span_equals_gm_elems(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["pv"])
        hs = p["head_stride_elems"]
        last = p["logical"]["batch"] - 1
        self.assertEqual((last + 1) * hs["a"], p["gm_elems"]["a"])
        self.assertEqual((last + 1) * hs["b"], p["gm_elems"]["b"])
        self.assertEqual((last + 1) * hs["c"], p["gm_elems"]["c"])

    def test_offsets_reject_bad_head_and_tile_indices(self):
        p = m.plan_tiling(*m.TARGET_SHAPES["score"])
        with self.assertRaises(m.BatchGemmError):
            m.gm_offsets(p, 8, 0, 0, 0)
        with self.assertRaises(m.BatchGemmError):
            m.gm_offsets(p, 0, 4, 0, 0)
        with self.assertRaises(m.BatchGemmError):
            m.gm_offsets(p, 0, 0, 16, 0)
        with self.assertRaises(m.BatchGemmError):
            m.gm_offsets(p, 0, 0, 0, 1)

    def test_pv_c_offset_start_and_k_offsets(self):
        p = m.plan_tiling(8, 512, 2048, 128)
        z = m.gm_offsets(p, 0, 0, 0, 0)
        self.assertEqual(z["a"], 0)
        self.assertEqual(z["b"], 0)
        self.assertEqual(z["c"], 0)
        # second K block of head0 tile(0,0): advances by kb1*m_pad*16  in A
        kb1 = p["tiles"]["kb"] // 16
        k1_off = m.gm_offsets(p, 0, 0, 0, 1)
        self.assertEqual(k1_off["a"], kb1 * p["padded"]["m_pad"] * 16)
        self.assertEqual(k1_off["b"], kb1 * p["logical"]["n"] * 16)


class DmaContractTests(unittest.TestCase):
    def test_dma_strides_unchanged_from_round83_formulas(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            mb, nb, kb = p["tiles"]["mb"], p["tiles"]["nb"], p["tiles"]["kb"]
            m_pad, n = p["padded"]["m_pad"], p["logical"]["n"]
            d = p["dma"]
            self.assertEqual(d["a_load"]["burst"], mb)
            self.assertEqual(d["a_load"]["src_stride"], m_pad - mb)
            self.assertEqual(d["a_load"]["nburst"], kb // 16)
            self.assertEqual(d["b_load"]["burst"], nb)
            self.assertEqual(d["b_load"]["src_stride"], n - nb)
            self.assertEqual(d["c_store"]["burst"], 2 * mb)
            self.assertEqual(d["c_store"]["dst_stride"], 2 * (m_pad - mb))
            self.assertEqual(d["c_store"]["nburst"], nb // 16)
            self.assertEqual(d["l0c_to_ub"]["burst_len_kb"], mb * nb * 4 // 1024)

    def test_dma_strides_are_non_negative(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            self.assertGreaterEqual(p["dma"]["a_load"]["src_stride"], 0)
            self.assertGreaterEqual(p["dma"]["b_load"]["src_stride"], 0)
            self.assertGreaterEqual(p["dma"]["c_store"]["dst_stride"], 0)


class HostLayoutTests(unittest.TestCase):
    """Verify pack/unpack index arithmetic in pure python (no torch needed)."""

    def test_pack_w_places_each_head_densely_and_distinctly(self):
        h, n, k = 2, 4, 32
        w = [[[float(hh * 1000 + ni * 100 + ki) for ki in range(k)]
              for ni in range(n)] for hh in range(h)]
        flat = ref_pack_w(w, h, n, k)
        k1 = k // 16
        self.assertEqual(len(flat), h * k1 * n * 16)
        # spot-check: head1 row3 k=17 -> k1=1,k0=1
        idx = (((1 * k1 + 1) * n + 3) * 16 + 1)
        self.assertEqual(flat[idx], w[1][3][17])
        # head0 and head1 must not alias
        idx0 = (((0 * k1 + 1) * n + 3) * 16 + 1)
        self.assertNotEqual(flat[idx0], flat[idx])

    def test_pack_a_zero_fills_only_the_padded_m_rows(self):
        h, mm, k = 2, 3, 32
        m_pad = 128
        a = [[[float(hh * 1000 + mi * 10 + ki) for ki in range(k)]
              for mi in range(mm)] for hh in range(h)]
        flat = ref_pack_a(a, h, mm, k, m_pad)
        k1 = k // 16
        self.assertEqual(len(flat), h * k1 * m_pad * 16)
        # real rows are present
        idx = (((0 * k1 + 0) * m_pad + 2) * 16 + 5)
        self.assertEqual(flat[idx], a[0][2][5])
        # padded rows 3..127 are exactly zero and no head's padding leaks
        for hh in range(h):
            for k1i in range(k1):
                for mi in range(mm, m_pad):
                    base = ((hh * k1 + k1i) * m_pad + mi) * 16
                    self.assertEqual(flat[base:base + 16], [0.0] * 16)

    def test_pack_unpack_roundtrip_preserves_values_and_head_order(self):
        h, mm, nn, k = 2, 3, 4, 32
        m_pad = 128
        a = [[[float(hh * 10000 + mi * 100 + ki) for ki in range(k)]
              for mi in range(mm)] for hh in range(h)]
        packed_a = ref_pack_a(a, h, mm, k, m_pad)
        # C is produced in (H,N1,Mp,16) layout with N=nn
        n1 = nn // 16 if nn % 16 == 0 else nn // 16 + 1
        c_flat = [0.0] * (h * n1 * m_pad * 16)
        for hh in range(h):
            for n1i in range(n1):
                for mi in range(m_pad):
                    for n0 in range(16):
                        idx = (((hh * n1 + n1i) * m_pad + mi) * 16 + n0)
                        c_flat[idx] = float(idx % 251)
        out = ref_unpack_c(c_flat, h, n1, m_pad, m_out=mm)
        self.assertEqual(len(out), h)
        self.assertEqual(len(out[0]), mm)
        # round-trip of the packed A values through the same index math
        for hh in range(h):
            for mi in range(mm):
                for ki in range(k):
                    k1i, k0 = divmod(ki, 16)
                    idx = (((hh * (k // 16) + k1i) * m_pad + mi) * 16 + k0)
                    self.assertEqual(packed_a[idx], a[hh][mi][ki])

    def test_unpack_c_crops_padded_rows_only(self):
        h, n1, m_pad, m_out = 2, 8, 512, 507
        c_flat = [1.0] * (h * n1 * m_pad * 16)
        out = ref_unpack_c(c_flat, h, n1, m_pad, m_out)
        self.assertEqual(len(out[0]), 507)
        self.assertEqual(len(out[0][0]), n1 * 16)

    def test_pack_helpers_reject_wrong_dims_and_dtypes_without_torch(self):
        """The guards must fire before any torch work; here we only check the dims."""
        with self.assertRaises(m.BatchGemmError):
            m.pack_a([[1, 2], [3, 4]])       # 2-D, not (H,M,K)
        with self.assertRaises(m.BatchGemmError):
            m.pack_w([[1, 2], [3, 4]])       # 2-D, not (H,N,K)
        with self.assertRaises(m.BatchGemmError):
            m.unpack_c([[1.0]], m=1)         # 2-D, not (H,N1,Mp,16)


class StaticKInitTests(unittest.TestCase):
    def test_k_loop_is_static_and_only_first_block_initialises(self):
        self.assertIn("for kt in range(k_tiles):", SOURCE)
        self.assertIn("init_l1out=(kt == 0)", SOURCE)
        # no dynamic loop over the K blocks (init_l1out must be a Python bool)
        self.assertNotIn("for_range(0, k_tiles)", SOURCE)
        self.assertNotIn("for_range(0, self._k", SOURCE)

    def test_one_single_l0c_tensor_outside_the_k_loop(self):
        self.assertIn('name="c_l0c", scope=tik.scope_cc', SOURCE)
        tree = ast.parse(SOURCE)
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_emit_kernel")
        c0_assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "c0" for t in n.targets)]
        self.assertEqual(len(c0_assigns), 1, "c0 must be allocated exactly once per task")

    def test_dynamic_iteration_is_only_the_outer_tile_step(self):
        self.assertIn("with t.for_range(0, steps) as step:", SOURCE)
        self.assertIn("with t.for_range(0, cores, block_num=cores) as core:", SOURCE)
        self.assertIn("with t.if_scope(task < tasks):", SOURCE)

    def test_each_k_block_has_a_war_sync_scope(self):
        self.assertIn("with t.new_stmt_scope(disable_sync=False):", SOURCE)
        self.assertIn("t.data_move(al, ag[a_off]", SOURCE)
        self.assertIn("t.matmul(c0, al, bl, mb, kb, nb,", SOURCE)


class HeadFoldingTests(unittest.TestCase):
    def test_head_is_folded_into_the_flat_task_not_a_loop_dimension(self):
        self.assertIn("head = task // per_head_tasks", SOURCE)
        self.assertIn("mt_idx = local // n_tiles", SOURCE)
        self.assertIn("nt_idx = local - mt_idx * n_tiles", SOURCE)
        # no per-head python loop in the emitted kernel
        self.assertNotIn("for head in range(", SOURCE)

    def test_head_stride_is_added_to_a_b_and_c(self):
        for token in ("head_a = head * hs[\"a\"]", "head_b = head * hs[\"b\"]",
                      "head_c = head * hs[\"c\"]"):
            self.assertIn(token, SOURCE)
        self.assertIn("a_off = head_a +", SOURCE)
        self.assertIn("b_off = head_b +", SOURCE)
        self.assertIn("c_off = head_c +", SOURCE)

    def test_gm_tensors_carry_the_batch_dimension(self):
        self.assertIn('(batch, k1, m_pad, 16), name="a_gm"', SOURCE)
        self.assertIn('(batch, k1, n, 16), name="b_gm"', SOURCE)
        self.assertIn('(batch, n1, m_pad, 16), name="c_gm"', SOURCE)


class ApiAndMetadataTests(unittest.TestCase):
    def test_public_api_and_aliases(self):
        for name in ("build_kernel", "plan_tiling", "pack_a", "pack_w", "pack_p",
                     "unpack_c", "capability_report", "flatten_task", "gm_offsets",
                     "kernel_name_for", "check_reference_r83_source"):
            self.assertTrue(callable(getattr(m, name)), name)

    def test_kernel_name_namespace_encodes_all_dimensions(self):
        p = m.plan_tiling(8, 512, 2048, 128)
        name = m.kernel_name_for(p)
        self.assertTrue(name.startswith("batch_gemm85_h8_m512_k2048_n128_"))
        self.assertIn("_mb128_nb128_kb128_c30", name)

    def test_two_different_batches_never_share_a_kernel_name(self):
        n1 = m.kernel_name_for(m.plan_tiling(8, 512, 128, 2048))
        n2 = m.kernel_name_for(m.plan_tiling(4, 512, 128, 2048))
        self.assertNotEqual(n1, n2)

    def test_five_gm_argument_abi_is_preserved(self):
        p = m.plan_tiling(*m.SUGGESTED_SMOKE_GEOMETRY)
        self.assertEqual(len(p["kernel_arg_order"]), 5)
        self.assertEqual(p["kernel_arg_order"],
                         ["a_gm_float16", "b_gm_float16", "aux0_gm_float16",
                          "aux1_gm_float16", "c_gm_float32"])
        self.assertIn("inputs=[gm[\"a\"], gm[\"b\"], gm[\"aux0\"], gm[\"aux1\"]]", SOURCE)

    def test_accumulation_metadata_forbids_half_partials(self):
        for shape in m.TARGET_SHAPES.values():
            acc = m.plan_tiling(*shape)["accumulation"]
            self.assertTrue(acc["single_l0c_across_all_k_blocks"])
            self.assertTrue(acc["init_first_k_block_only"])
            self.assertFalse(acc["half_partial"])
            self.assertFalse(acc["fp32_ub_partial_sum"])
            self.assertEqual(acc["final_output_dtype"], "float32")

    def test_batched_metadata_requires_distinct_per_head_b(self):
        p = m.plan_tiling(*m.SUGGESTED_SMOKE_GEOMETRY)
        self.assertTrue(p["batched"]["per_head_distinct_b"])
        self.assertTrue(p["batched"]["head_offset_applied_to_a_b_c"])
        self.assertEqual(p["batched"]["task_flatten"],
                         "head*(m_tiles*n_tiles) + m_tile*n_tiles + n_tile")

    def test_no_fp32_weight_shadow_or_bias_operand_in_plan(self):
        self.assertNotIn("fp32_weight", json_dumps_plan())
        self.assertNotIn("bias", json_dumps_plan())


def json_dumps_plan():
    import json
    return json.dumps({k: m.plan_tiling(*v) for k, v in m.TARGET_SHAPES.items()})


class HeadLeakSimulationTests(unittest.TestCase):
    """End-to-end pure-python check that each head reads only its own B matrix.

    We pack synthetic per-head weights whose values encode (head, n, k), then
    walk the *kernel's own* offset arithmetic to gather the B tile each task
    would cube-multiply. If any head's tile contained another head's tag, the
    head stride or the task flattening would be wrong.
    """

    def _simulate(self, batch, mm, k, n):
        p = m.plan_tiling(batch, mm, k, n)
        k1 = p["padded"]["k1"]
        mb, nb, kb = p["tiles"]["mb"], p["tiles"]["nb"], p["tiles"]["kb"]
        kb1 = kb // 16
        # tag[i] = (head, n, k) packed exactly as pack_w would lay it out
        tag = [None] * p["gm_elems"]["b"]
        for hh in range(batch):
            for ni in range(n):
                for ki in range(k):
                    k1i, k0 = divmod(ki, 16)
                    idx = (((hh * k1 + k1i) * n + ni) * 16 + k0)
                    tag[idx] = (hh, ni, ki)
        seen_heads = set()
        for task in range(p["grid"]["tasks"]):
            head, mt, nt = m.flatten_task(task, batch, p["grid"]["n_tiles"],
                                          p["grid"]["per_head_tasks"])
            seen_heads.add(head)
            for kt in range(p["accumulation"]["k_tiles"]):
                off = m.gm_offsets(p, head, mt, nt, kt)
                for bi in range(kb1):
                    for ni_local in range(nb):
                        base = off["b"] + bi * n * 16 + ni_local * 16
                        values = tag[base:base + 16]
                        self.assertNotIn(None, values,
                                         f"head{head} task{task} read an unpacked B slot")
                        for (th, tn, tk) in values:
                            self.assertEqual(th, head,
                                             "head leakage: task read another head's B")
                            self.assertEqual(tn, nt * nb + ni_local)
                            self.assertEqual(tk, kt * kb + bi * 16 + (tk % 16))
        self.assertEqual(seen_heads, set(range(batch)))

    def test_score_shaped_small_batch_reads_only_own_head(self):
        self._simulate(batch=2, mm=128, k=128, n=256)

    def test_pv_shaped_small_batch_reads_only_own_head(self):
        self._simulate(batch=2, mm=128, k=256, n=128)

    def test_three_heads_never_alias(self):
        self._simulate(batch=3, mm=128, k=128, n=128)


class CStoreBoundsTests(unittest.TestCase):
    def test_every_c_store_stays_inside_its_head_slice(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            c_elems = (p["tiles"]["nb"] // 16) * p["tiles"]["mb"] * 16
            hs = p["head_stride_elems"]["c"]
            for head in range(p["logical"]["batch"]):
                for mt in range(p["grid"]["m_tiles"]):
                    for nt in range(p["grid"]["n_tiles"]):
                        off = m.gm_offsets(p, head, mt, nt, 0)["c"]
                        self.assertGreaterEqual(off, head * hs)
                        self.assertLessEqual(off + c_elems, (head + 1) * hs)

    def test_c_offsets_are_16_element_aligned(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            for head in range(p["logical"]["batch"]):
                for mt in range(p["grid"]["m_tiles"]):
                    for nt in range(p["grid"]["n_tiles"]):
                        off = m.gm_offsets(p, head, mt, nt, 0)
                        self.assertEqual(off["a"] % 16, 0)
                        self.assertEqual(off["b"] % 16, 0)
                        self.assertEqual(off["c"] % 16, 0)


class KernelIdentityTests(unittest.TestCase):
    def test_all_target_shapes_have_unique_kernel_names(self):
        names = [m.kernel_name_for(m.plan_tiling(*s)) for s in m.TARGET_SHAPES.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_last_chunk_and_full_chunk_are_distinct_kernels(self):
        full = m.kernel_name_for(m.plan_tiling(8, 512, 128, 2048))
        last = m.kernel_name_for(m.plan_tiling(8, 507, 128, 2048))
        self.assertNotEqual(full, last)

    def test_name_encodes_batch_and_does_not_reuse_round83_prefix(self):
        name = m.kernel_name_for(m.plan_tiling(8, 512, 128, 2048))
        self.assertTrue(name.startswith("batch_gemm85_"))
        self.assertNotIn("blocked_gemm83", name)
        self.assertNotIn("blocked_dynamic83", name)

    def test_steps_cover_every_task_exactly_once_in_round_robin_order(self):
        for shape in m.TARGET_SHAPES.values():
            p = m.plan_tiling(*shape)
            cores, steps, tasks = p["grid"]["cores"], p["grid"]["steps"], p["grid"]["tasks"]
            scheduled = []
            for core in range(cores):
                for step in range(steps):
                    task = core + step * cores
                    if task < tasks:
                        scheduled.append(task)
            self.assertEqual(sorted(scheduled), list(range(tasks)))
            self.assertEqual(len(scheduled), len(set(scheduled)))


class PackRoundtripAgainstPlanTests(unittest.TestCase):
    def test_packed_a_size_matches_plan_for_all_shapes(self):
        for shape in m.TARGET_SHAPES.values():
            h, mm, k, n = shape
            p = m.plan_tiling(*shape)
            k1, m_pad = p["padded"]["k1"], p["padded"]["m_pad"]
            self.assertEqual(h * k1 * m_pad * 16, p["gm_elems"]["a"])
            n1 = p["padded"]["n1"]
            self.assertEqual(h * n1 * m_pad * 16, p["gm_elems"]["c"])

    def test_packed_b_size_matches_plan_for_all_shapes(self):
        for shape in m.TARGET_SHAPES.values():
            h, mm, k, n = shape
            p = m.plan_tiling(*shape)
            k1 = p["padded"]["k1"]
            self.assertEqual(h * k1 * n * 16, p["gm_elems"]["b"])

    def test_host_unpack_crop_leaves_exactly_the_pad_rows(self):
        for shape in ((8, 507, 128, 2048), (8, 507, 2048, 128)):
            p = m.plan_tiling(*shape)
            self.assertEqual(p["padded"]["m_pad"] - p["logical"]["m"], 5)
            self.assertEqual(p["padded"]["m_pad_rows"], 5)
            self.assertEqual(p["padded"]["m_pad"], 512)


class ReferencePinningTests(unittest.TestCase):
    def test_reference_check_rejects_a_modified_copy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "blocked_dynamic_steps83.py"
            fake.write_text("# not the executed source\n")
            with self.assertRaises(m.BatchGemmError):
                m.check_reference_r83_source(fake)

    def test_reference_check_accepts_the_real_executed_source(self):
        if not R83.is_file():
            self.skipTest("Round83 source not present")
        self.assertEqual(m.check_reference_r83_source(R83), m.PINNED_R83_SHA256)

    def test_r83_module_is_not_mutated_by_importing_this_one(self):
        if not R83.is_file():
            self.skipTest("Round83 source not present")
        before = hashlib.sha256(R83.read_bytes()).hexdigest()
        importlib.reload(m)
        after = hashlib.sha256(R83.read_bytes()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(after, m.PINNED_R83_SHA256)


class ManifestAndScopeTests(unittest.TestCase):
    def test_scope_directory_contains_no_npu_receipts(self):
        """This fork must not have produced runtime receipts; only source+tests."""
        for pattern in ("*.o", "*.json.result", "run_finished.json"):
            self.assertEqual(list(HERE.glob(pattern)), [], pattern)

    def test_no_torch_or_cann_binary_artefacts(self):
        self.assertEqual(list(HERE.glob("kernel_meta")), [])
        self.assertEqual(list(HERE.glob("kernels")), [])


class PinnedCapabilityTests(unittest.TestCase):
    def test_pins_match_the_executed_round83_kernel(self):
        self.assertEqual(m.PINNED_COMPILE_SOC, "Ascend910B")
        self.assertEqual(m.PINNED_RUNTIME_SOC, "Ascend910B")
        self.assertEqual(m.PINNED_CORE_NUM, 30)
        self.assertEqual(m.PINNED_MAX_CORES, 30)
        self.assertEqual(m.PINNED_L1_SIZE, 1048576)
        self.assertEqual(m.PINNED_L0C_SIZE, 262144)
        self.assertEqual(m.PINNED_R83_SHA256,
                         "b5a06cfdb4c83a2d2f8c84eb56e8794f8a38d2732c4ff5e1fbef0901b520facb")

    def test_round83_reference_file_is_intact_on_disk(self):
        if not R83.is_file():
            self.skipTest("Round83 reference source not present in this checkout")
        check = m.check_reference_r83_source(R83)
        self.assertEqual(check, m.PINNED_R83_SHA256)
        # and the R83 source really does contain the arithmetic we copied
        text = R83.read_text()
        self.assertIn("init_l1out=(kt == 0)", text)
        self.assertIn("blocked_dynamic83_", text)

    def test_capability_report_guards_are_present_but_not_executed(self):
        self.assertIn("Intrinsic_mmad_f162f32", SOURCE)
        self.assertIn("block_num=cores", SOURCE)
        self.assertIn("CORE_NUM", SOURCE)
        self.assertIn("must not be faked", SOURCE)

    def test_no_unsafe_delete_or_move_of_computed_paths(self):
        for forbidden in ("rmtree", "os.remove", "os.unlink", "shutil.move",
                          "os.rmdir", ".unlink("):
            self.assertNotIn(forbidden, SOURCE, f"unsafe path operation {forbidden}")

    def test_build_workdirs_are_unique_and_artefacts_copied(self):
        self.assertIn("tempfile.mkdtemp(prefix=\"batch85_build_\"", SOURCE)
        self.assertIn("shutil.copyfile(built_obj, obj)", SOURCE)
        self.assertIn("shutil.copyfile(built_meta, meta_path)", SOURCE)
        self.assertIn("work.parent != out_root", SOURCE)

    def test_compile_lock_and_refuse_overwrite(self):
        self.assertIn("fcntl.flock(lock.fileno(), fcntl.LOCK_EX)", SOURCE)
        self.assertIn(".compile.lock", SOURCE)
        self.assertIn("refusing to overwrite existing kernel directory", SOURCE)

    def test_object_sha_is_recorded_and_cached_binary_is_revalidated(self):
        self.assertIn("hashlib.sha256(obj.read_bytes()).hexdigest()", SOURCE)
        self.assertIn("refusing to reuse a mismatched binary", SOURCE)
        self.assertIn('"obj_sha256"', SOURCE)
        self.assertIn('"kernel_symbol"', SOURCE)
        self.assertIn('"dst_dtype"', SOURCE)
        self.assertIn('"block_dim"', SOURCE)


class SourceOnlyTests(unittest.TestCase):
    def test_module_scope_is_stdlib_only(self):
        tree = ast.parse(SOURCE)
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"torch", "tbe", "numpy", "aiohttp", "requests", "socket"}
        self.assertEqual(imported & forbidden, set())

    def test_tbe_and_torch_are_imported_lazily_inside_functions(self):
        self.assertIn("from tbe import tik", SOURCE)
        self.assertIn("import torch", SOURCE)
        # module-level ast has no such import
        tree = ast.parse(SOURCE)
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        text = "\n".join(ast.get_source_segment(SOURCE, n) or "" for n in top)
        self.assertNotIn("tbe", text)
        self.assertNotIn("torch", text)

    def test_no_ssh_subprocess_or_network(self):
        for forbidden in ("subprocess", "ssh", "socket", "urllib", "http"):
            self.assertNotIn(forbidden, SOURCE)

    def test_module_declares_source_only_and_no_numeric_claim(self):
        self.assertIn("SOURCE ONLY", SOURCE)
        self.assertIn('"executed": False', SOURCE)
        self.assertIn('"not_a_speed_claim": True', SOURCE)
        self.assertIn('"full_encoder_precision_accepted": False', SOURCE)
        self.assertIn('"no_fp32_weight_shadow": True', SOURCE)
        self.assertIn('"per_head_distinct_b_required": True', SOURCE)

    def test_does_not_edit_or_import_for_mutation_the_round83_source(self):
        # R83 is referenced by path+hash only, never imported for its builder
        self.assertNotIn("import blocked_dynamic_steps83", SOURCE)
        self.assertNotIn("from blocked_dynamic_steps83", SOURCE)
        self.assertIn("check_reference_r83_source", SOURCE)

    def test_manifest_source_only_flags_when_present(self):
        path = HERE / "manifest85.json"
        if not path.is_file():
            self.skipTest("manifest not written yet")
        import json
        manifest = json.loads(path.read_text())
        self.assertFalse(manifest.get("compiled"))
        self.assertFalse(manifest.get("executed_on_npu"))
        self.assertFalse(manifest.get("proves_precision"))
        self.assertFalse(manifest.get("proves_speed"))


if __name__ == "__main__":
    unittest.main()