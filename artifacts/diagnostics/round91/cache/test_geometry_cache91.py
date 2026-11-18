#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round91 vision geometry cache.

No torch, no torch_npu, no NPU, no model and no SSH: every operand is a stdlib
double that implements exactly the duck-typed attributes the cache reads, so this
file can run anywhere. These tests prove structure, gating, raw-bit primed
identity, fail-closed signature/grid/parameter validation, restore and the
non-claims; they are NOT an accuracy or speed result.
"""

import ast
import hashlib
import importlib.util
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "geometry_cache91.py"
_CANDIDATE = (ROOT.parent.parent / "round79" / "candidate" / "candidate_encoder79.py")
_spec = importlib.util.spec_from_file_location("geometry_cache91_under_test", MODULE_PATH)
cache91 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache91)

TOKENS = 8064
POS_BYTES = TOKENS * 1152 * 2
ROT_BYTES = TOKENS * 36 * 4


class _Dtype:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


class _Device:
    def __init__(self, kind, index):
        self.type, self.index = kind, index


class _Array:
    """Deterministic raw bytes for a fake tensor (no numpy)."""

    def __init__(self, seed, size):
        self._seed, self._size = bytes(seed), int(size)

    def tobytes(self):
        if self._size == 0:
            return b""
        repeats = self._size // len(self._seed) + 1
        return (self._seed * repeats)[: self._size]


class _Tensor:
    """The minimal torch-tensor surface the cache is allowed to use."""

    _next_ptr = 0x1000

    def __init__(self, shape, dtype, seed=b"\x01\x02\x03\x04", device=("npu", 0),
                 values=None, version=0, ptr=None, offset=0):
        self.shape = tuple(int(x) for x in shape)
        self._dtype, self._seed = dtype, bytes(seed)
        self.device = _Device(*device)
        self._values = values
        self._version, self._offset = int(version), int(offset)
        if ptr is None:
            ptr = _Tensor._next_ptr
            _Tensor._next_ptr += 0x1000
        self._ptr = int(ptr)

    # -- metadata ---------------------------------------------------------
    @property
    def dtype(self):
        return _Dtype(self._dtype)

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def element_size(self):
        return {"torch.float16": 2, "torch.float32": 4, "torch.int64": 8}[self._dtype]

    def data_ptr(self):
        return self._ptr

    def storage_offset(self):
        return self._offset

    def stride(self):
        strides, running = [], 1
        for size in reversed(self.shape):
            strides.append(running)
            running *= max(size, 1)
        return tuple(reversed(strides))

    def detach(self):
        return _Tensor(self.shape, self._dtype, self._seed, (self.device.type, self.device.index),
                       self._values, self._version, self._ptr, self._offset)

    def contiguous(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return _Array(self._seed, self.numel() * self.element_size())

    def clone(self):
        return _Tensor(self.shape, self._dtype, self._seed, (self.device.type, self.device.index),
                       self._values, self._version)

    def tolist(self):
        if self._values is None:
            raise AssertionError("tolist() called on a tensor without values")
        return self._values

    # -- test-only mutators ----------------------------------------------
    def bump_version(self):
        self._version += 1
        return self

    def set_seed(self, seed):
        self._seed = bytes(seed)
        return self

    def set_values(self, values):
        self._values = values
        return self

    def raw_sha(self):
        return hashlib.sha256(self.numpy().tobytes()).hexdigest()


def _grid_bytes(values):
    """Canonical little-endian int64 payload of a grid, matching the real fixture bytes."""
    return struct.pack("<%dq" % (len(values) * 3),
                       *[int(v) for row in values for v in row])


def _grid(values=((1, 48, 84), (1, 48, 84)), **kwargs):
    kwargs.setdefault("seed", _grid_bytes(values))
    return _Tensor((len(values), 3), "torch.int64", values=[list(row) for row in values], **kwargs)


class _Rotary:
    def __init__(self, seed=b"\x11" * 8):
        self.dim, self.theta = 36, 10000.0
        self._inv_freq = _Tensor((18,), "torch.float32", seed=seed, device=("cpu", None))


class _FakeVision:
    """Class-level geometry operators, mirroring the reviewed candidate."""

    def __init__(self, config=None, weight_seed=b"\x21" * 8, rotary=None):
        self.config = config or SimpleNamespace(hidden_size=1152, num_heads=16,
                                                intermediate_size=4304,
                                                num_position_embeddings=2304,
                                                spatial_merge_size=2)
        self.pos_embed = SimpleNamespace(
            weight=_Tensor((2304, 1152), "torch.float16", seed=weight_seed))
        self.rotary_pos_emb = rotary or _Rotary()
        self.num_grid_per_side, self.spatial_merge_size = 48, 2
        self.original_calls = {"position": 0, "rotary": 0}

    def _seed(self, grid, extra):
        payload = json.dumps(grid.tolist(), sort_keys=True).encode() + extra
        return hashlib.sha256(payload).digest()

    def fast_pos_embed_interpolate(self, grid):
        self.original_calls["position"] += 1
        return _Tensor((TOKENS, 1152), "torch.float16",
                       seed=self._seed(grid, self.pos_embed.weight._seed))

    def rot_pos_emb(self, grid):
        self.original_calls["rotary"] += 1
        return _Tensor((TOKENS, 36), "torch.float32",
                       seed=self._seed(grid, self.rotary_pos_emb._inv_freq._seed))


class _Base(unittest.TestCase):
    def make(self, **kwargs):
        vision = _FakeVision(**kwargs)
        cache = cache91.VisionGeometryCache91(vision, candidate_path=_CANDIDATE)
        return vision, cache

    def primed(self, **kwargs):
        vision, cache = self.make(**kwargs)
        cache.prime(_grid())
        cache.set_enabled(True)
        return vision, cache


class StructureTests(_Base):
    def test_patches_exactly_the_two_geometry_methods_per_instance(self):
        vision = _FakeVision()
        before = set(vision.__dict__)
        cache = cache91.VisionGeometryCache91(vision, candidate_path=_CANDIDATE)
        added = set(vision.__dict__) - before
        self.assertEqual(added, {cache91.POSITION_METHOD, cache91.ROTARY_METHOD})
        for name in (cache91.POSITION_METHOD, cache91.ROTARY_METHOD):
            bound = getattr(vision, name)
            self.assertIs(bound.__self__, vision)
            self.assertIs(bound.__func__, cache._wrappers[name].__func__)
        self.assertEqual(cache.snapshot()["patched_method_names"],
                         [cache91.POSITION_METHOD, cache91.ROTARY_METHOD])
        self.assertTrue(cache.snapshot()["per_instance_patch_only"])

    def test_refuses_instance_attribute_already_present(self):
        vision = _FakeVision()
        vision.rot_pos_emb = lambda grid: None
        with self.assertRaises(cache91.GeometryCacheError):
            cache91.VisionGeometryCache91(vision, candidate_path=_CANDIDATE)

    def test_refuses_non_real_structure(self):
        cases = [
            dict(config=SimpleNamespace(hidden_size=1024, num_heads=16,
                                        num_position_embeddings=2304,
                                        spatial_merge_size=2)),
            dict(config=SimpleNamespace(hidden_size=1152, num_heads=16,
                                        num_position_embeddings=1024,
                                        spatial_merge_size=2)),
        ]
        for kwargs in cases:
            with self.assertRaises(cache91.GeometryCacheError):
                cache91.VisionGeometryCache91(_FakeVision(**kwargs), candidate_path=_CANDIDATE)
        vision = _FakeVision()
        vision.num_grid_per_side = 32
        with self.assertRaises(cache91.GeometryCacheError):
            cache91.VisionGeometryCache91(vision, candidate_path=_CANDIDATE)
        vision = _FakeVision()
        vision.pos_embed.weight = _Tensor((2304, 1152), "torch.bfloat16")
        with self.assertRaises(cache91.GeometryCacheError):
            cache91.VisionGeometryCache91(vision, candidate_path=_CANDIDATE)

    def test_structure_report_pins_half_weight_and_float_inv_freq(self):
        vision, cache = self.make()
        report = cache.snapshot()["structure"]
        self.assertEqual(report["pos_embed_weight_dtype"], "torch.float16")
        self.assertEqual(report["inv_freq"]["dtype"], "torch.float32")
        self.assertEqual(report["rotary_dim"], 36)
        self.assertEqual(report["rotary_theta"], 10000.0)
        self.assertEqual(report["token_count"], TOKENS)
        self.assertEqual(report["position_dim"], 1152)
        self.assertEqual(report["rotary_flatten_dim"], 36)


class PrimeTests(_Base):
    def test_prime_calls_originals_twice_and_verifies_raw_bits(self):
        vision, cache = self.make()
        snapshot = cache.prime(_grid())
        self.assertEqual(vision.original_calls, {"position": 2, "rotary": 2})
        self.assertTrue(snapshot["primed"])
        self.assertFalse(snapshot["enabled"])
        for name, shape, dtype in ((cache91.POSITION_METHOD, [TOKENS, 1152], "torch.float16"),
                                   (cache91.ROTARY_METHOD, [TOKENS, 36], "torch.float32")):
            row = snapshot["cached"][name]
            self.assertEqual(row["shape"], shape)
            self.assertEqual(row["dtype"], dtype)
            self.assertTrue(row["sha256_unchanged_since_prime"])
            self.assertEqual(row["prime_identity"]["raw_bytes_equal"], True)
        self.assertEqual(snapshot["extra_bytes"], POS_BYTES + ROT_BYTES)
        self.assertEqual(snapshot["grid_values"], [[1, 48, 84], [1, 48, 84]])
        self.assertEqual(snapshot["grid_raw_sha256"], cache91.REAL_GRID_SHA256)
        self.assertEqual(snapshot["grid_raw_sha256"],
                         "6a43f833fe86f2b0ae8abf21273d8b739d07df42a6b4c8195a96712e00116a4e")

    def test_prime_checks_the_recorded_grid_bytes_not_just_the_values(self):
        vision, cache = self.make()
        # Values read back as the real fixture, but the raw payload is not the recorded
        # 48 canonical int64 bytes: the pinned fixture hash must reject it.
        tampered = _grid(seed=b"\x00" * 48)
        with self.assertRaises(cache91.GeometryCacheError):
            cache.prime(tampered)
        self.assertFalse(cache.snapshot(verify_cache=False)["primed"])

    def test_prime_rejects_non_real_grid_and_disabled_reprime_allowed(self):
        vision, cache = self.make()
        with self.assertRaises(cache91.GeometryCacheError):
            cache.prime(_grid(((1, 48, 84), (1, 48, 80))))
        cache.prime(_grid())
        cache.set_enabled(True)
        with self.assertRaises(cache91.GeometryCacheError):
            cache.prime(_grid())
        cache.set_enabled(False)
        cache.prime(_grid())
        self.assertEqual(vision.original_calls["position"], 4)

    def test_prime_never_modifies_the_input_grid(self):
        vision, cache = self.make()
        grid = _grid()
        before = (cache91.tensor_signature(grid), cache91.grid_values(grid))
        cache.prime(grid)
        cache.set_enabled(True)
        vision.fast_pos_embed_interpolate(grid)
        vision.rot_pos_emb(grid)
        self.assertEqual(before, (cache91.tensor_signature(grid), cache91.grid_values(grid)))

    def test_enable_before_prime_fails_closed(self):
        vision, cache = self.make()
        with self.assertRaises(cache91.GeometryCacheError):
            cache.set_enabled(True)
        self.assertFalse(cache.snapshot()["enabled"])
        with self.assertRaises(cache91.GeometryCacheError):
            cache.set_enabled("yes")


class EnabledPathTests(_Base):
    def test_disabled_path_is_the_exact_original(self):
        vision, cache = self.make()
        grid = _grid()
        out = vision.fast_pos_embed_interpolate(grid)
        out_rot = vision.rot_pos_emb(grid)
        self.assertEqual(vision.original_calls, {"position": 1, "rotary": 1})
        expected = _FakeVision().fast_pos_embed_interpolate(grid)
        self.assertEqual(out.raw_sha(), expected.raw_sha())
        snapshot = cache.snapshot(verify_cache=False)
        self.assertEqual(snapshot["stats"]["disabled_calls"], 2)
        self.assertEqual(snapshot["stats"]["hits"][cache91.POSITION_METHOD], 0)

    def test_enabled_returns_the_same_cached_objects_without_recomputing(self):
        vision, cache = self.primed()
        grid = _grid()
        calls_before = dict(vision.original_calls)
        position = vision.fast_pos_embed_interpolate(grid)
        rotary = vision.rot_pos_emb(grid)
        self.assertEqual(vision.original_calls, calls_before)
        self.assertIs(position, cache._cache[cache91.POSITION_METHOD]["tensor"])
        self.assertIs(rotary, cache._cache[cache91.ROTARY_METHOD]["tensor"])
        self.assertIs(vision.fast_pos_embed_interpolate(grid), position)
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["stats"]["hits"],
                         {cache91.POSITION_METHOD: 2, cache91.ROTARY_METHOD: 1})
        self.assertEqual(snapshot["stats"]["grid_fast_hits"], 2)
        self.assertTrue(snapshot["cached_geometry_only"])
        self.assertFalse(snapshot["caches_pixels_patch_hidden_deepstack_or_text"])
        self.assertEqual(snapshot["float_weights"], 0)

    def test_new_identical_grid_object_is_accepted_by_value(self):
        vision, cache = self.primed()
        fresh = _grid()
        self.assertNotEqual(fresh.data_ptr(), cache._grid_object.data_ptr())
        position = vision.fast_pos_embed_interpolate(fresh)
        self.assertIs(position, cache._cache[cache91.POSITION_METHOD]["tensor"])
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]["grid_value_rereads"], 1)

    def test_unversioned_inference_grid_rereads_values_every_call(self):
        vision, cache = self.primed()
        inner = _grid()
        class InferenceGrid:
            def __getattr__(self, name):
                if name == '_version':
                    raise RuntimeError('Inference tensors do not track version counter.')
                return getattr(inner, name)
            def detach(self): return self
            def cpu(self): return self
        grid = InferenceGrid()
        self.assertIsNone(cache91.tensor_signature(grid)['version'])
        vision.fast_pos_embed_interpolate(grid)
        vision.rot_pos_emb(grid)
        stats = cache.snapshot(verify_cache=False)['stats']
        self.assertEqual(stats['grid_value_rereads'], 2)
        self.assertEqual(stats['grid_fast_hits'], 0)
        inner.set_values([[1,48,84],[1,44,84]])
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(grid)

    def test_new_grid_with_different_values_fails_closed(self):
        vision, cache = self.primed()
        with self.assertRaises(cache91.GeometryCacheError):
            vision.fast_pos_embed_interpolate(_grid(((1, 48, 84), (1, 44, 84))))
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]["validation_failures"], 1)

    def test_same_object_version_bump_rereads_values_instead_of_trusting_pointer(self):
        vision, cache = self.primed()
        grid = cache._grid_object
        grid.bump_version()
        position = vision.fast_pos_embed_interpolate(grid)
        self.assertIs(position, cache._cache[cache91.POSITION_METHOD]["tensor"])
        stats = cache.snapshot(verify_cache=False)["stats"]
        self.assertEqual(stats["grid_value_rereads"], 1)
        self.assertEqual(stats["grid_fast_hits"], 0)

    def test_same_object_mutated_values_fails_closed(self):
        vision, cache = self.primed()
        grid = cache._grid_object
        # A real in-place write also bumps torch's _version, so the fast path is skipped.
        grid.set_values([[1, 48, 84], [1, 40, 84]]).bump_version()
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(grid)

    def test_recycled_pointer_of_a_new_object_never_serves_stale_geometry(self):
        vision, cache = self.primed()
        # Simulate allocator reuse: the new object carries the old data_ptr AND version,
        # so only object identity (not the pointer) distinguishes it from the primed grid.
        recycled = _grid(ptr=cache._grid_object.data_ptr(),
                         version=cache._grid_signature["version"])
        self.assertIsNot(recycled, cache._grid_object)
        self.assertEqual(cache91.tensor_signature(recycled), cache._grid_signature)
        # Identical values are still accepted, but only after the values were re-read.
        self.assertIs(vision.fast_pos_embed_interpolate(recycled),
                      cache._cache[cache91.POSITION_METHOD]["tensor"])
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]["grid_value_rereads"], 1)
        # A recycled pointer with different values can never serve the cached tensors.
        different = _grid(((1, 48, 84), (1, 44, 84)),
                          ptr=cache._grid_object.data_ptr(),
                          version=cache._grid_signature["version"])
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(different)


class DriftTests(_Base):
    def test_changed_position_weight_signature_fails_closed(self):
        vision, cache = self.primed()
        vision.pos_embed.weight.bump_version()
        with self.assertRaises(cache91.GeometryCacheError):
            vision.fast_pos_embed_interpolate(_grid())

    def test_changed_inv_freq_values_fail_closed_even_without_a_version_bump(self):
        vision, cache = self.primed()
        vision.rotary_pos_emb._inv_freq.set_seed(b"\x77" * 8)
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(_grid())

    def test_replaced_inv_freq_tensor_fails_closed(self):
        vision, cache = self.primed()
        vision.rotary_pos_emb._inv_freq = _Tensor((18,), "torch.float32", seed=b"\x55" * 8,
                                                  device=("cpu", None))
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(_grid())

    def test_changed_theta_or_grid_side_or_config_fails_closed(self):
        vision, cache = self.primed()
        vision.rotary_pos_emb.theta = 5000.0
        with self.assertRaises(cache91.GeometryCacheError):
            vision.rot_pos_emb(_grid())
        vision, cache = self.primed()
        vision.num_grid_per_side = 40
        with self.assertRaises(cache91.GeometryCacheError):
            vision.fast_pos_embed_interpolate(_grid())
        vision, cache = self.primed()
        vision.config.hidden_size = 1024
        with self.assertRaises(cache91.GeometryCacheError):
            vision.fast_pos_embed_interpolate(_grid())


class RestoreTests(_Base):
    def test_uninstall_restores_pristine_and_is_idempotent(self):
        vision, cache = self.primed()
        snapshot = cache.uninstall()
        self.assertTrue(snapshot["uninstalled"])
        for name in (cache91.POSITION_METHOD, cache91.ROTARY_METHOD):
            self.assertNotIn(name, vision.__dict__)
        calls = dict(vision.original_calls)
        vision.rot_pos_emb(_grid())
        self.assertEqual(vision.original_calls["rotary"], calls["rotary"] + 1)
        cache.uninstall()

    def test_uninstall_refuses_to_clobber_a_third_party_patch(self):
        vision, cache = self.make()
        cache.prime(_grid())
        vision.rot_pos_emb = lambda grid: None
        with self.assertRaises(cache91.GeometryCacheError):
            cache.uninstall()


class SourceAndNonClaimTests(unittest.TestCase):
    def test_source_pins_match_the_reviewed_candidate_and_methods(self):
        status = cache91.reference_status(_CANDIDATE)
        self.assertTrue(status["matches_pins"])
        self.assertEqual(status["candidate_sha256"], cache91.PINNED_CANDIDATE_SHA256)
        for name in (cache91.POSITION_METHOD, cache91.ROTARY_METHOD):
            self.assertEqual(status["method_sha256"][name], cache91.PINNED_METHOD_SHA256[name])
        self.assertIn("geometry only", cache91.CACHE_SCOPE)

    def test_missing_class_in_another_file_is_rejected(self):
        with self.assertRaises(cache91.GeometryCacheError):
            cache91.method_source_digest(MODULE_PATH, "NoSuchVisionModel", "rot_pos_emb")
        with self.assertRaises(cache91.GeometryCacheError):
            cache91.reference_status(ROOT / "does_not_exist_91.py")

    def test_module_scope_imports_are_stdlib_only_and_torch_is_never_imported(self):
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertLessEqual(imported, {"ast", "hashlib", "json", "pathlib", "types", "__future__"})
        # No torch/torch_npu/TIK import statement anywhere in the module (the docstring
        # mentions the words, so this checks the AST, not a substring).
        module_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_names.add(node.module.split(".")[0])
        self.assertFalse(module_names & {"torch", "torch_npu", "tbe", "numpy"})

    def test_snapshot_declares_no_precision_or_speed_result(self):
        vision, cache = _Base().primed()
        snapshot = cache.snapshot()
        self.assertIsNone(snapshot["numerical_tolerance"])
        self.assertFalse(snapshot["full_encoder_precision_accepted"])
        self.assertFalse(snapshot["whole_goal_complete"])
        self.assertEqual(snapshot["float_weights"], 0)
        self.assertFalse(snapshot["parameter_dtype_changed"])
        self.assertFalse(snapshot["patches_torch_global"])
        self.assertEqual(set(snapshot["cached"]),
                         {cache91.POSITION_METHOD, cache91.ROTARY_METHOD})

    def test_snapshot_detects_a_consumer_mutating_a_cached_tensor(self):
        vision, cache = _Base().primed()
        self.assertTrue(cache.snapshot()["cached"][cache91.POSITION_METHOD]
                        ["sha256_unchanged_since_prime"])
        cache._cache[cache91.ROTARY_METHOD]["tensor"].set_seed(b"\x99" * 8)
        self.assertFalse(cache.snapshot()["cached"][cache91.ROTARY_METHOD]
                         ["sha256_unchanged_since_prime"])

    def test_raw_identity_is_bit_equality_not_a_value_tolerance(self):
        a = _Tensor((2, 2), "torch.float16", seed=b"\x00\x01")
        b = _Tensor((2, 2), "torch.float16", seed=b"\x00\x01")
        c = _Tensor((2, 2), "torch.float16", seed=b"\x00\x02")
        self.assertTrue(cache91.raw_identity(a, b)["raw_bytes_equal"])
        self.assertFalse(cache91.raw_identity(a, c)["raw_bytes_equal"])


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_the_delivered_files_and_stays_non_claiming(self):
        manifest = json.loads((ROOT / "manifest91.json").read_text())
        for name, row in manifest["files"].items():
            data = (ROOT / name).read_bytes()
            self.assertEqual(row["sha256"], hashlib.sha256(data).hexdigest(), name)
            self.assertEqual(row["bytes"], len(data), name)
        self.assertEqual(manifest["compiled"], False)
        self.assertEqual(manifest["executed_on_npu"], False)
        self.assertFalse(manifest["proves_precision"])
        self.assertFalse(manifest["proves_speed"])
        self.assertEqual(manifest["compiles_or_runs_anything"], False)
        self.assertEqual(manifest["patched_methods"],
                         [cache91.POSITION_METHOD, cache91.ROTARY_METHOD])
        self.assertEqual(manifest["cache_scope"], cache91.CACHE_SCOPE)
        self.assertEqual(manifest["grid_raw_sha256"], cache91.REAL_GRID_SHA256)
        self.assertEqual(manifest["candidate_sha256"], cache91.PINNED_CANDIDATE_SHA256)


if __name__ == "__main__":
    unittest.main()