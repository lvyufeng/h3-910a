#!/usr/bin/env python3
"""Static contract tests for the Round82 FP16->FP32 GEMM prototype.

Standard library only: no torch, no torch_npu, no CANN import, no device. These
tests run on a machine without the accelerator and therefore cannot and do not
claim any numerical or performance result.

What they CAN establish: the prototype's declared dtype/geometry contract is
self-consistent, the SDK receipts really hash what they claim, the launch path is
the proven legacy one and not the known-broken aclrtBinary family, and nothing in
the deliverables quietly claims to have executed.
"""

import ast
import hashlib
import json
from pathlib import Path
import py_compile
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gemm_fp16_fp32out82 as proto  # noqa: E402  (import is stdlib-safe)


def source(name):
    return (HERE / name).read_text(encoding="utf-8")


class ProvenanceTests(unittest.TestCase):
    def test_prototype_modules_compile(self):
        for name in ("gemm_fp16_fp32out82.py", "run_gemm_probe82.py"):
            py_compile.compile(str(HERE / name), doraise=True)

    def test_sdk_receipts_hash_the_files_they_name(self):
        receipts = json.loads((HERE / "sdk_receipts82.json").read_text())
        self.assertEqual(len(receipts["files"]), 13)
        self.assertFalse(receipts["executed"])
        self.assertFalse(receipts["proves_precision"])
        for entry in receipts["files"]:
            path = HERE / entry["local_copy"]
            self.assertTrue(path.is_file(), entry["local_copy"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             entry["sha256"], entry["local_copy"])
            self.assertEqual(path.stat().st_size, entry["bytes"])
            self.assertIn("/python/site-packages/", entry["remote_path"])

    def test_key_sdk_rows_are_really_present_in_the_copied_source(self):
        params = (HERE / "sdk" / "te_platform_cce_params.py").read_text()
        self.assertIn('"matmul": {', params)
        self.assertIn('ASCEND_910AIC: ["s8s8s32", "u8s8s32", "f16f16f32"]', params)
        self.assertIn('ASCEND_910BAIC: ["s8s8s32", "u8s8s32", "f16f16f32"]', params)
        # The decisive negative: no half-output float combination is listed.
        self.assertNotIn('"f16f16f16"', params)
        common = (HERE / "sdk" / "te_tik_api_cube_cube_common.py").read_text()
        self.assertIn("is_910b_soc()", common)
        self.assertIn("bias.dtype, dst.dtype", common)
        cube = (HERE / "sdk" / "te_tik_api_cube_tik_cube_api.py").read_text()
        self.assertIn("dst: A tensor of type float16, float32, or int32", cube)
        api = (HERE / "sdk" / "api.py").read_text()
        self.assertIn('dst_dtype="float16"', api)
        self.assertIn('support "float16" "float32"', api)


class ContractTests(unittest.TestCase):
    def test_two_variants_and_their_output_dtypes(self):
        self.assertEqual(sorted(proto.VARIANTS), ["fp32_bias_fp16_out", "fp32_out"])
        self.assertEqual(proto.VARIANTS["fp32_out"]["dst_dtype"], "float32")
        self.assertFalse(proto.VARIANTS["fp32_out"]["use_bias"])
        self.assertEqual(proto.VARIANTS["fp32_bias_fp16_out"]["dst_dtype"], "float16")
        self.assertTrue(proto.VARIANTS["fp32_bias_fp16_out"]["use_bias"])

    def test_plan_fits_one_tile_within_the_documented_limits(self):
        plan = proto.build_plan("fp32_out")
        lim = plan["byte_sizes"]
        # L1_SIZE 1048576 and L0C_SIZE 262144 are the values recorded in
        # artifacts/design/ascendc_flash_attn/TBE_BUILD.md for this part.
        self.assertLessEqual(lim["a_float16"] + lim["b_float16"], 1048576)
        self.assertLessEqual(lim["out_float32"], 262144)
        self.assertEqual(plan["compile_soc"], "Ascend910B")
        self.assertFalse(plan["executed"])

    def test_bias_variant_uses_fp32_bias_and_three_inputs(self):
        plan = proto.build_plan("fp32_bias_fp16_out")
        self.assertTrue(plan["uses_fp32_bias"])
        self.assertEqual(plan["byte_sizes"]["bias_float32"], proto.MICRO_N * 4)
        self.assertEqual(plan["kernel_arg_order"],
                         ["a_gm_float16", "b_gm_float16", "bias_gm_float32",
                          "out_gm_<dst_dtype>"])

    def test_compile_soc_must_be_the_real_device_name_not_910A(self):
        # bias is gated on is_910b_soc(), which reads the compile SoC; forcing
        # Ascend910A would silently disable the bias operand.
        self.assertEqual(proto.PINNED_COMPILE_SOC, "Ascend910B")
        self.assertEqual(proto.PINNED_COMPILE_SOC, proto.PINNED_DEVICE_SOC)
        self.assertNotEqual(proto.PINNED_COMPILE_SOC, "Ascend910A")

    def test_h3_geometry_is_documented_but_not_claimed_buildable(self):
        self.assertEqual(proto.H3_GEOMETRY["kv_proj"], {"M": 2043, "K": 5120, "N": 128})
        # The single-tile route needs A and B resident in the 1 MiB L1; the real
        # KV projection operand alone is ~20 MiB of float16, so it cannot be one
        # tile and requires blocked tiling.
        kv = proto.H3_GEOMETRY["kv_proj"]
        a_bytes = kv["M"] * kv["K"] * 2
        b_bytes = kv["N"] * kv["K"] * 2
        self.assertGreater(a_bytes, 1048576)
        self.assertGreater(a_bytes + b_bytes, 1048576)
        text = " ".join(source("gemm_fp16_fp32out82.py").split())
        self.assertIn("needs explicit blocked tiling", text)
        self.assertIn("does not fit", text)

    def test_build_never_claims_execution(self):
        text = source("gemm_fp16_fp32out82.py")
        self.assertIn('"executed": False', text)
        self.assertIn("SOURCE ONLY", text)


class LaunchPathTests(unittest.TestCase):
    def test_launcher_uses_the_proven_legacy_runtime_not_aclrtbinary(self):
        c = source("gemm_launcher82.c")
        for symbol in ("rtDevBinaryRegister", "rtFunctionRegister", "rtKernelLaunch"):
            self.assertIn(symbol, c)
        for broken in ("aclrtBinaryLoadFromFile(", "aclrtBinaryGetFunction(",
                       "aclrtLaunchKernelWithHostArgs("):
            self.assertNotIn(broken, c)
        self.assertIn("RT_DEV_BINARY_MAGIC_ELF", c)
        # The 507015 failure mode must be documented in the source itself.
        self.assertIn("507015", c)

    def test_launch_arity_helper_matches_kernel_arg_order(self):
        c = source("gemm_launcher82.c")
        self.assertIn("void* args[3] = { a, b, out }", c)
        self.assertIn("void* args[4] = { a, b, bias, out }", c)

    def test_probe_refuses_to_overwrite_evidence_and_is_explicit_about_scope(self):
        text = source("run_gemm_probe82.py")
        self.assertIn("refusing to overwrite existing evidence", text)
        self.assertIn('"not_h3_geometry": True', text)
        self.assertIn('"not_a_speed_claim": True', text)
        self.assertIn('"full_encoder_precision_accepted": False', text)
        self.assertIn("not the H3 geometry", text)

    def test_probe_builds_golden_from_the_same_fp16_operands(self):
        text = source("run_gemm_probe82.py")
        self.assertIn("a_np.astype(np.float32) @ b_np.astype(np.float32).T", text)
        self.assertIn("golden = golden + bias_np[None, :]", text)
        # The native-ordering control must exist for the biased variant.
        self.assertIn("vs_native_half_bias_ordering", text)


if __name__ == "__main__":
    unittest.main()