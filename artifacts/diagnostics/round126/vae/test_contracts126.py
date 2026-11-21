"""Portable R126 tests. No Torch, model, remote access or kernel execution."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from contracts126 import (AUDIO_LATENT_SHAPE, ContractError, LATENTS_BYTES, LATENTS_SHA256,
                          VIDEO_LATENT_SHAPE, VIDEO_OUTPUT_SHAPE, aggregate_rank_receipts,
                          atomic_json, is_baseline_receipt, validate_manifest)
from decode_only126 import EventRegions, Journal, ParameterRewriteRejected, observed_installer, parser

ROOT = Path(__file__).resolve().parent
SHA = "a" * 64


def manifest():
    roles = {name: {"import": "example." + name, "path": "/tmp/source_" + name + ".py", "sha256": SHA}
             for name in ("adapter", "optimizations", "dispatch", "sdpa_shim", "nan_compat", "activation")}
    roles["dispatch"]["import"] = "vllm_omni.diffusion.models.minimax_h3.ops.vae.dispatch"
    return {"schema_version": 1, "approved_executor": "lead", "approved_root": "/tmp/h3_perf12601", "world_size": 8,
            "checkpoint": "/models/FL2VA", "latents": "/tmp/latents.pt", "nan_compat_dir": "/tmp/compat",
            "latents_sha256": LATENTS_SHA256, "latents_bytes": LATENTS_BYTES,
            "video_latent_shape": VIDEO_LATENT_SHAPE, "audio_latent_shape": AUDIO_LATENT_SHAPE,
            "latent_dtype": "torch.float32", "parameter_dtype": "torch.float32", "video_autocast_dtype": "torch.float16", "hf32": False,
            "torch_version": "2.10.0+cpu", "torch_npu_version": "2.10.0.post7", "video_config_sha256": SHA, "audio_config_sha256": SHA, "video_source_config_sha256": SHA,
            "modules": roles, "component_source_sha256": {"video_vae": {"klvae.py": SHA}, "audio_vae": {"minimax_h3_audio_vae.py": SHA}},
            "native_source_sha256": {name: SHA for name in ("native_video", "native_decoder", "native_attention", "native_blocks", "native_func", "native_flash", "native_parallel", "native_audio")},
            "environment": {"VLLM_OMNI_910A_FLASH_ATTN": "1", "VLLM_OMNI_910A_NO_FUSED_ACT": "1"},
            "video_output_shape": VIDEO_OUTPUT_SHAPE, "video_output_dtype": "torch.uint8", "audio_output_shape": [1, 2, 165600], "audio_output_dtype": "torch.float32",
            "initialization_model_scope": "video_audio_vae_only_no_DiT_no_encoder_forward"}


class ManifestTests(unittest.TestCase):
    def test_valid_and_no_torch_import(self):
        self.assertIs(validate_manifest(manifest())["hf32"], False)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("torch_npu", sys.modules)

    def test_precision_source_and_dispatch_fail_closed(self):
        for field, value in (("parameter_dtype", "torch.float16"), ("latent_dtype", "torch.bfloat16"),
                             ("video_autocast_dtype", "torch.bfloat16"), ("latents_sha256", "b" * 64),
                             ("hf32", True), ("world_size", 4)):
            data = manifest(); data[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_manifest(data)
        data = manifest(); data["modules"]["dispatch"]["import"] = "vllm_omni.diffusion.models.minimax_h3.ops.dispatch"
        with self.assertRaises(ContractError):
            validate_manifest(data)

    def test_unsupported_fused_and_old_hooks_rejected(self):
        for key in ("VLLM_OMNI_910A_VAE_FINITE_SELECT", "H3_NPU_ALIGNMENT124_BOOTSTRAP", "H3_NPU_SAVE_FINAL_LATENTS124", "H3_NPU_CAPTURE_NATIVE_MEDIA124"):
            data = manifest(); data["environment"][key] = "1"
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_manifest(data)
        data = manifest(); data["environment"]["VLLM_OMNI_910A_NO_FUSED_ACT"] = "0"
        with self.assertRaises(ContractError):
            validate_manifest(data)

    def test_default_real_installer_observation(self):
        args = parser().parse_args(["--checkpoint", "/m", "--latents", "/l", "--root", "/r", "--label", "run126"])
        self.assertEqual(args.installer_mode, "observe-native")
        self.assertFalse(args.execute)


class BindingTests(unittest.TestCase):
    def test_journal_removes_only_own_instance_binding(self):
        class Owner:
            def f(self):
                return "original"
        owner = Owner(); journal = Journal()
        journal.bind(owner, "f", lambda: "wrapped")
        self.assertEqual(owner.f(), "wrapped")
        journal.close()
        self.assertEqual(owner.f(), "original")
        self.assertNotIn("f", owner.__dict__)

    def test_unknown_concurrent_binding_not_blindly_restored(self):
        owner = SimpleNamespace(f=lambda: 1); journal = Journal()
        journal.bind(owner, "f", lambda: 2)
        other = lambda: 3
        owner.f = other
        with self.assertRaises(ContractError):
            journal.close()
        self.assertIs(owner.f, other)

    def test_no_sync_or_CPU_copy_in_profile_wrapper(self):
        calls = []
        class FakeEvent:
            def __init__(self, **kwargs):
                pass
            def record(self):
                calls.append("record")
            def elapsed_time(self, end):
                return 10.0
        npu = SimpleNamespace(Event=FakeEvent, synchronize=lambda: calls.append("sync"))
        regions = EventRegions(SimpleNamespace(npu=npu), 2, names=("region",))
        owner = SimpleNamespace(f=lambda x: x + 1)
        regions.bind(owner, "f", "region")
        calls.clear()
        self.assertEqual(owner.f(1), 2)
        self.assertEqual(owner.f(2), 3)
        self.assertEqual(owner.f(3), 4)  # Overflow still completes native work.
        self.assertNotIn("sync", calls)
        proof = regions.receipt()
        self.assertTrue(proof["incomplete_profile"])
        self.assertEqual(proof["regions"]["region"]["calls"], 3)
        regions.close()
        self.assertEqual(owner.f(1), 2)

    def test_installer_bypass_is_not_native_false(self):
        class Param:
            dtype = "FP32"
            device = "cpu"
            def numel(self): return 10
            def element_size(self): return 4
            def is_inference(self): return False
            def is_floating_point(self): return True
        decoder = SimpleNamespace(parameters=lambda: [Param()])
        count = []
        def native(decoder, *, device):
            count.append(device); return False
        adapter = SimpleNamespace(install_h3_vae_optimizations=native)
        record = {"calls": 0, "native_return": None}
        with observed_installer(SimpleNamespace(float32="FP32"), adapter, "preserve-fp32", record):
            self.assertFalse(adapter.install_h3_vae_optimizations(decoder, device="npu:0"))
        self.assertEqual(count, [])
        self.assertIsNone(record["native_return"])
        self.assertIs(record["native_called"], False)
        self.assertIs(adapter.install_h3_vae_optimizations, native)

    def test_native_false_is_observed_and_half_rewrite_rejected(self):
        class Param:
            dtype = "FP32"
            device = "cpu"
            def numel(self): return 10
            def element_size(self): return 4
            def is_inference(self): return False
            def is_floating_point(self): return True
        parameter = Param(); decoder = SimpleNamespace(parameters=lambda: [parameter])
        adapter = SimpleNamespace(install_h3_vae_optimizations=lambda decoder, *, device: False)
        record = {"calls": 0, "native_return": None}
        with observed_installer(SimpleNamespace(float32="FP32"), adapter, "observe-native", record):
            self.assertFalse(adapter.install_h3_vae_optimizations(decoder, device="npu:0"))
        self.assertIs(record["native_return"], False)
        self.assertIs(record["native_called"], True)
        def rewrite(decoder, *, device):
            parameter.dtype = "Half"; return True
        adapter.install_h3_vae_optimizations = rewrite
        record = {"calls": 0, "native_return": None}
        with observed_installer(SimpleNamespace(float32="FP32"), adapter, "observe-native", record):
            with self.assertRaises(ParameterRewriteRejected):
                adapter.install_h3_vae_optimizations(decoder, device="npu:0")
        self.assertIs(record["native_return"], True)
        self.assertIs(record["forward_executed_after_parameter_rewrite"], False)
        self.assertEqual(parameter.dtype, "Half")  # Never pretend a widening restores it.


class ReceiptTests(unittest.TestCase):
    def test_native_noop_baseline_and_active_candidate(self):
        row = {"installer_mode": "observe-native", "installer": {"native_called": True, "native_return": False},
               "binding_snapshot": {"installed_marker": False, "blocks": []}, "parameter_contract": {"all_float_parameters_fp32": True}}
        self.assertTrue(is_baseline_receipt(row))
        row["installer"]["native_return"] = True
        self.assertFalse(is_baseline_receipt(row))

    def test_rank_max_precedes_median_and_cold_excluded(self):
        rows = []
        for rank in range(8):
            warm = [1.0, 1.0, 1.0]
            if rank < 3: warm[rank] = 100.0
            calls = [{"call_index": i, "wall_s": wall, "stream_s": wall / 2, "output": {"sha256": SHA}, "comparison": None}
                     for i, wall in enumerate([999.0] + warm)]
            rows.append({"rank": rank, "completed": True, "latents_sha256": LATENTS_SHA256, "driver_sha256": SHA, "manifest_sha256": SHA,
                         "installer_mode": "observe-native", "installer": {"native_return": False},
                         "parameter_contract": {"all_float_parameters_fp32": True}, "cases": {"video_clean": calls}})
        summary = aggregate_rank_receipts(rows)
        self.assertEqual(summary["cases"]["video_clean"]["median_warm_rank_max_wall_s"], 100.0)
        self.assertEqual(summary["cases"]["video_clean"]["cold_rank_max_wall_s"], 999.0)
        self.assertFalse(summary["accepted_lossless"])
        bad = copy.deepcopy(rows); bad[-1]["rank"] = 0
        with self.assertRaises(ContractError): aggregate_rank_receipts(bad)
        bad = copy.deepcopy(rows); bad[-1]["cleanup_failure"] = [{"operation": "offload"}]
        with self.assertRaises(ContractError): aggregate_rank_receipts(bad)

    def test_atomic_publication_and_no_symlink_overwrite(self):
        # All temporary writes stay in the sole assigned scope; resolve it before
        # the TemporaryDirectory cleanup deletes this exact private test tree.
        with tempfile.TemporaryDirectory(prefix="test126-", dir=ROOT) as td:
            directory = Path(td).resolve(strict=True)
            self.assertEqual(directory.parent, ROOT)
            self.assertTrue(directory.name.startswith("test126-"))
            path = directory / "receipt.json"
            atomic_json(path, {"one": 1})
            atomic_json(path, {"two": 2})
            self.assertEqual(json.loads(path.read_text()), {"two": 2})
            link = directory / "link.json"; link.symlink_to(path)
            with self.assertRaises(ContractError): atomic_json(link, {"bad": True})

    def test_native_parallel_overrides_do_not_instantiate_generic_executor(self):
        source = ROOT.parent / "native_sources" / "vae.py"
        tree = ast.parse(source.read_text())
        video = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE")
        methods = {node.name: node for node in video.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertIn("process_group", [arg.arg for arg in methods["set_parallel_size"].args.args])
        self.assertIn("is_distributed_enabled", methods)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "init_distributed" for node in ast.walk(video)))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "DistributedVaeExecutor" for node in ast.walk(video)))
        driver = ast.parse((ROOT / "decode_only126.py").read_text())
        setters = [node for node in ast.walk(driver) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "set_parallel_size"]
        self.assertEqual(len(setters), 1)
        group = next(keyword.value for keyword in setters[0].keywords if keyword.arg == "process_group")
        self.assertEqual(ast.unparse(group), "torch.distributed.group.WORLD")

    def test_lazy_imports_and_no_hidden_generation_calls(self):
        tree = ast.parse((ROOT / "decode_only126.py").read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(getattr(node, "module", "") == "torch" or any(alias.name in ("torch", "torch_npu") for alias in getattr(node, "names", [])) for node in top_imports))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr in ("generate", "encode_image", "encode_video", "encode_prompt") for node in calls))
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "set_default_dtype" and node.args and isinstance(node.args[0], ast.Attribute) and node.args[0].attr == "float16" for node in calls))


if __name__ == "__main__":
    unittest.main()
