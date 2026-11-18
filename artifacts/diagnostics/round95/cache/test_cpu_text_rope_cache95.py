#!/usr/bin/env python3
"""Stdlib-only contract tests for the Round95 original-HF CPU Half RoPE cache.

No torch, no torch_npu, no HF model, no NPU and no SSH: every tensor is a stdlib
double implementing exactly the duck-typed attributes the cache reads. These
tests prove provenance pins, config gates, install/restore identity, per-call
gating (version/identity/pointer/value), cache-mutation refusal and the
non-claims. They are NOT an accuracy or speed result: the real CPU table is only
produced when the root agent primes the cache on the NPU host.
"""
import ast
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent
DIAGNOSTICS = ROOT.parent.parent
CANDIDATE = DIAGNOSTICS / "round79" / "candidate" / "candidate_encoder79.py"
HF_COPY = DIAGNOSTICS / "round86" / "modeling_qwen3_vl_original86.py"
MODULE_PATH = ROOT / "cpu_text_rope_cache95.py"

_spec = importlib.util.spec_from_file_location("cpu_text_rope_cache95_under_test", MODULE_PATH)
cache95 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cache95)

if "candidate_encoder79" not in sys.modules:
    _module = type(sys)("candidate_encoder79")
    _module.__file__ = str(CANDIDATE)
    sys.modules["candidate_encoder79"] = _module

SEQ = 2043
HIDDEN = 5120
HEADS = 64
HEAD_DIM = 128
DIM = HEAD_DIM
HALF_BYTES = 2
POSITION_VALUES = b"fixture-positions-"


class _DType:
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


class _Device:
    def __init__(self, kind, index):
        self.type, self.index = kind, index


class _Array:
    def __init__(self, data):
        self._data = bytes(data)

    def tobytes(self):
        return self._data


class _Tensor:
    """Minimal duck-typed tensor: metadata + deterministic raw bytes."""

    _next_ptr = 0x100000

    def __init__(self, shape, dtype, device=("npu", 0), seed=b"\x01\x02\x03\x04",
                 version=0, ptr=None, offset=0, unversioned=False, reads=None):
        self.shape = tuple(int(v) for v in shape)
        self._dtype, self._device = dtype, _Device(*device)
        self._seed = bytes(seed)
        self._v = int(version)
        self._unversioned = bool(unversioned)
        self._offset = int(offset)
        self._ptr = self._next_ptr if ptr is None else int(ptr)
        _Tensor._next_ptr += 0x1000
        self.reads = [] if reads is None else reads

    @property
    def dtype(self):
        return _DType(self._dtype)

    @property
    def device(self):
        return self._device

    @property
    def _version(self):
        if self._unversioned:
            raise RuntimeError("Inference tensors do not track version counter.")
        return self._v

    def bump_version(self):
        self._v += 1
        return self

    def data_ptr(self):
        return self._ptr

    def storage_offset(self):
        return self._offset

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def element_size(self):
        return {"torch.float16": 2, "torch.float32": 4, "torch.int64": 8}[self._dtype]

    def stride(self):
        stride, running = [], 1
        for size in reversed(self.shape):
            stride.append(running)
            running *= max(int(size), 1)
        return tuple(reversed(stride))

    def _raw(self):
        size = self.numel() * self.element_size()
        if size == 0:
            return b""
        repeats = size // len(self._seed) + 1
        return (self._seed * repeats)[:size]

    def detach(self):
        return _Tensor(self.shape, self._dtype, (self._device.type, self._device.index),
                       self._seed, self._v, self._ptr, self._offset, self._unversioned,
                       self.reads)

    def cpu(self):
        return _Tensor(self.shape, self._dtype, ("cpu", None), self._seed, self._v,
                       self._ptr, self._offset, self._unversioned, self.reads)

    def contiguous(self):
        return self

    def clone(self):
        return _Tensor(self.shape, self._dtype, (self._device.type, self._device.index),
                       self._seed, self._v, None, self._offset, self._unversioned,
                       self.reads)

    def numpy(self):
        self.reads.append(1)
        return _Array(self._raw())


def hidden(device=("npu", 0), unversioned=False):
    return _Tensor((1, SEQ, HIDDEN), "torch.float16", device,
                   seed=b"hidden-dummy", unversioned=unversioned)


def positions(seed=POSITION_VALUES, device=("npu", 0), unversioned=False,
              ptr=None, version=0, reads=None):
    return _Tensor((3, 1, SEQ), "torch.int64", device, seed=seed, version=version,
                   ptr=ptr, unversioned=unversioned, reads=reads)


def table(name, device=("npu", 0)):
    seed = b"cos-table" if name == "cos" else b"sin-table"
    return _Tensor((1, SEQ, HEAD_DIM), "torch.float16", device, seed=seed)


def config(**overrides):
    values = dict(rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                   "mrope_section": [24, 20, 20]},
                  max_position_embeddings=262144, hidden_size=HIDDEN,
                  num_attention_heads=HEADS, head_dim=HEAD_DIM)
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeRotary:
    def __init__(self, cfg=None):
        self.config = cfg if cfg is not None else config()
        self.dim, self.base, self.mrope_section = DIM, 10000.0, [24, 20, 20]
        self.original_calls = []
        self.inv = _Tensor((DIM // 2,), "torch.float32", ("cpu", None), seed=b"inv-freq")

    def forward(self, x, position_ids):
        self.original_calls.append((x, position_ids))
        return table("cos"), table("sin")

    def _get_inv_freq(self, device):
        return self.inv

    def parameters(self):
        return []


_FakeRotary.__module__ = "candidate_encoder79"
_FakeRotary.__name__ = "MiniMaxH3Qwen3VLTextRotaryEmbedding"


def make_cache(**rotary_overrides):
    rotary = _FakeRotary(**rotary_overrides)
    return rotary, cache95.CPUTextRopeCache95(rotary, candidate_path=CANDIDATE)


def prime_like(cache, position_ids=None, cos=None, sin=None, x=None):
    """Populate the primed state exactly as prime() would, without torch/HF."""
    x = x if x is not None else hidden()
    position_ids = position_ids if position_ids is not None else positions()
    cos = cos if cos is not None else table("cos")
    sin = sin if sin is not None else table("sin")
    cache._structure = cache95.config_fingerprint(cache._rotary)
    cache._hf_source_path = HF_COPY
    cache._x_metadata = {"shape": list(cache95.EXPECTED_HIDDEN_SHAPE),
                         "dtype": cache95.EXPECTED_STATE_DTYPE,
                         "device_type": "npu", "device_index": 0}
    cache._positions_object = position_ids
    cache._positions_signature = cache95.tensor_signature(position_ids)
    cache._positions_bytes = cache95.raw_bytes(position_ids)
    cache._cache = {}
    for name, tensor in (("cos", cos), ("sin", sin)):
        cache._cache[name] = {"tensor": tensor,
                              "signature": cache95.tensor_signature(tensor),
                              "sha256": cache95.tensor_raw_sha(tensor),
                              "shape": list(cache95.EXPECTED_TABLE_SHAPE),
                              "dtype": cache95.EXPECTED_STATE_DTYPE,
                              "bytes": cache95.EXPECTED_TABLE_BYTES // 2}
    cache._cpu_table = {"repeat_bits_equal": True,
                        "matches_R94_measured_CPU_table": True}
    cache._native_probe = {"cos_vs_CPU_table": {"raw_bits_equal": False},
                           "sin_vs_CPU_table": {"raw_bits_equal": False}}
    cache._prime_count = 1
    cache._prime_seconds = 0.0
    cache._primed = True
    return x, position_ids, cos, sin


class PinContract(unittest.TestCase):
    def test_candidate_and_reference_pins_match_committed_files(self):
        self.assertEqual(hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
                         cache95.PINNED_CANDIDATE_SHA256)
        self.assertEqual(hashlib.sha256(HF_COPY.read_bytes()).hexdigest(),
                         cache95.PINNED_HF_SOURCE_SHA256)
        digests = cache95.method_source_digests(CANDIDATE, cache95.ROTARY_CLASS_NAME)
        for name, expected in cache95.PINNED_CANDIDATE_METHOD_SHA256.items():
            self.assertEqual(digests[name], expected, name)

    def test_r78_position_and_r94_table_pins_are_committed_evidence(self):
        r92 = (DIAGNOSTICS / "round92" / "cpu" / "original_cpu_block25_92.py").read_text()
        self.assertIn(cache95.PINNED_POSITION_IDS_SHA256, r92)
        r94 = json.loads((DIAGNOSTICS / "round94" / "rope94.json").read_text())
        text = r94["text_actual_Half"]
        self.assertEqual(text["cos_vs_CPU"]["cpu"]["sha256"], cache95.PINNED_CPU_COS_SHA256)
        self.assertEqual(text["sin_vs_CPU"]["cpu"]["sha256"], cache95.PINNED_CPU_SIN_SHA256)
        self.assertEqual(r94["text_inv_freq_CPU_vs_candidate"]["cpu"]["sha256"],
                         cache95.PINNED_CPU_INV_FREQ_SHA256)
        self.assertEqual(text["cos_vs_CPU"]["npu"]["sha256"],
                         cache95.R94_NATIVE_OBSERVED_SHA256["cos"])
        self.assertEqual(text["sin_vs_CPU"]["npu"]["sha256"],
                         cache95.R94_NATIVE_OBSERVED_SHA256["sin"])
        self.assertFalse(text["cos_vs_CPU"]["raw_bits_equal"])
        self.assertEqual(cache95.EXPECTED_TABLE_BYTES, 2 * SEQ * HEAD_DIM * HALF_BYTES)


class ProvenanceGate(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=str(ROOT)))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_wrong_candidate_file_is_refused(self):
        copied = self.tmp / "candidate_encoder79.py"
        copied.write_bytes(CANDIDATE.read_bytes() + b"\n# drift\n")
        with self.assertRaises(cache95.TextRopeCacheError):
            cache95.CPUTextRopeCache95(_FakeRotary(), candidate_path=copied)

    def test_wrong_rotary_class_is_refused(self):
        class Other(_FakeRotary):
            pass

        with self.assertRaises(cache95.TextRopeCacheError):
            cache95.CPUTextRopeCache95(Other(), candidate_path=CANDIDATE)

    def test_config_gate_refuses_dynamic_rope_and_wrong_geometry(self):
        cases = [dict(cfg=config(rope_parameters={"rope_type": "dynamic",
                                                  "rope_theta": 10000.0,
                                                  "mrope_section": [24, 20, 20]})),
                 dict(cfg=config(rope_parameters={"rope_type": "default",
                                                  "rope_theta": 5000000.0,
                                                  "mrope_section": [24, 20, 20]}))]
        rotary, _ = make_cache()
        for overrides in cases:
            with self.assertRaises(cache95.TextRopeCacheError):
                cache95.CPUTextRopeCache95(_FakeRotary(**overrides),
                                           candidate_path=CANDIDATE)
        rotary.dim = 64
        with self.assertRaises(cache95.TextRopeCacheError):
            cache95.pinned_config_gate(rotary)
        rotary.dim = DIM
        rotary.mrope_section = [20, 20, 20]
        with self.assertRaises(cache95.TextRopeCacheError):
            cache95.pinned_config_gate(rotary)

    def test_constructor_installs_disabled_and_has_zero_teacher_state(self):
        rotary, cache = make_cache()
        self.assertTrue(cache._installed)
        self.assertFalse(cache._enabled)
        self.assertFalse(cache._primed)
        snapshot = cache.snapshot(verify_cache=False)
        self.assertTrue(snapshot["per_instance_patch_only"])
        self.assertFalse(snapshot["patches_torch_global"])
        self.assertFalse(snapshot["patches_candidate_class"])
        self.assertFalse(snapshot["patches_private_candidate_functions"])
        self.assertEqual(snapshot["teacher_weights"], 0)
        self.assertEqual(snapshot["cached"], {})
        self.assertTrue(snapshot["cached_rope_only"])
        self.assertIsNone(snapshot["numerical_tolerance"])


class DisabledAndStateMachine(unittest.TestCase):
    def test_disabled_path_delegates_to_the_exact_original(self):
        rotary, cache = make_cache()
        x, pos = hidden(), positions()
        out = rotary.forward(x, pos)
        self.assertEqual(len(rotary.original_calls), 1)
        self.assertTrue(rotary.original_calls[0][0] is x)
        self.assertTrue(rotary.original_calls[0][1] is pos)
        self.assertTrue(out[0] is not None)
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]["disabled_calls"], 1)

    def test_enable_requires_prime_and_bool(self):
        rotary, cache = make_cache()
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.set_enabled(True)
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.set_enabled("yes")

    def test_prime_refused_while_enabled_and_after_uninstall(self):
        rotary, cache = make_cache()
        prime_like(cache)
        cache.set_enabled(True)
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.prime(hidden(), positions())
        cache.set_enabled(False)
        cache.uninstall()
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.prime(hidden(), positions())


class PinnedPrimeInputs(unittest.TestCase):
    def test_position_fixture_gate_rejects_metadata_and_value_drift(self):
        rotary, cache = make_cache()
        good = positions(seed=POSITION_VALUES)
        # The real pinned R78 value is checked in PinContract; here the gate logic
        # is exercised by pinning the double's own hash for the accept path.
        fake_sha = cache95.tensor_raw_sha(good)
        with mock.patch.object(cache95, "PINNED_POSITION_IDS_SHA256", fake_sha):
            self.assertEqual(cache._position_fixture_gate(good), fake_sha)
            with self.assertRaises(cache95.TextRopeCacheError):
                cache._position_fixture_gate(positions(seed=b"another-sequence"))
        # Under the real pin the double is refused, never silently accepted.
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._position_fixture_gate(good)
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._position_fixture_gate(_Tensor((3, 1, SEQ - 1), "torch.int64",
                                                 seed=POSITION_VALUES))
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._position_fixture_gate(_Tensor((3, 1, SEQ), "torch.int32",
                                                 seed=POSITION_VALUES))
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._position_fixture_gate(positions(device=("cpu", None)))

    def test_x_metadata_gate_requires_half_npu_fixture_shape(self):
        rotary, cache = make_cache()
        cache._x_metadata_from(hidden())
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._x_metadata_from(_Tensor((1, SEQ, HIDDEN), "torch.bfloat16",
                                           seed=b"x"))
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._x_metadata_from(_Tensor((1, SEQ, HIDDEN), "torch.float16",
                                           device=("cpu", None), seed=b"x"))
        with self.assertRaises(cache95.TextRopeCacheError):
            cache._x_metadata_from(_Tensor((1, 7, HIDDEN), "torch.float16", seed=b"x"))


class EnabledDispatch(unittest.TestCase):
    def test_enabled_returns_the_same_cached_objects_without_recompute(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        first = rotary.forward(x, pos)
        second = rotary.forward(x, pos)
        self.assertTrue(first[0] is cos and first[1] is sin)
        self.assertTrue(second[0] is cos and second[1] is sin)
        self.assertEqual(rotary.original_calls, [])
        stats = cache.snapshot(verify_cache=False)["stats"]
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["position_fast_hits"], 2)
        self.assertEqual(stats["position_value_rereads"], 0)

    def test_new_equal_positions_object_is_value_validated(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        fresh = positions()
        self.assertNotEqual(fresh.data_ptr(), pos.data_ptr())
        out = rotary.forward(x, fresh)
        self.assertTrue(out[0] is cos)
        stats = cache.snapshot(verify_cache=False)["stats"]
        self.assertEqual(stats["position_value_rereads"], 1)
        self.assertEqual(stats["position_fast_hits"], 0)

    def test_inference_positions_are_value_validated_every_call(self):
        rotary, cache = make_cache()
        unversioned = positions(unversioned=True)
        x, pos, cos, sin = prime_like(cache, position_ids=unversioned)
        cache.set_enabled(True)
        rotary.forward(x, unversioned)
        rotary.forward(x, unversioned)
        stats = cache.snapshot(verify_cache=False)["stats"]
        self.assertEqual(stats["position_value_rereads"], 2)
        self.assertEqual(stats["position_fast_hits"], 0)

    def test_recycled_pointer_new_object_never_takes_the_fast_path(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        recycled = positions(ptr=pos.data_ptr(), version=pos._version)
        self.assertIsNot(recycled, pos)
        self.assertEqual(cache95.tensor_signature(recycled),
                         cache95.tensor_signature(pos))
        rotary.forward(x, recycled)
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]
                         ["position_value_rereads"], 1)

    def test_same_object_version_bump_forces_value_validation(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        pos.bump_version()
        rotary.forward(x, pos)
        stats = cache.snapshot(verify_cache=False)["stats"]
        self.assertEqual(stats["position_fast_hits"], 0)
        self.assertEqual(stats["position_value_rereads"], 1)

    def test_different_positions_values_are_refused(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, positions(seed=b"another-sequence"))
        self.assertEqual(cache.snapshot(verify_cache=False)["stats"]
                         ["validation_failures"], 1)

    def test_x_content_is_irrelevant_but_metadata_drift_is_refused(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        # Same metadata, different content: the table does not depend on x values.
        self.assertTrue(rotary.forward(
            _Tensor((1, SEQ, HIDDEN), "torch.float16", seed=b"other-hidden"), pos)[0] is cos)
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(_Tensor((1, SEQ, HIDDEN), "torch.float16",
                                   device=("npu", 1), seed=b"x"), pos)
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(_Tensor((1, 7, HIDDEN), "torch.float16", seed=b"x"), pos)

    def test_positions_device_index_drift_is_refused(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, positions(device=("npu", 1)))

    def test_mutated_cached_table_is_refused(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        cos.bump_version()
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, pos)
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.snapshot()

    def test_config_and_source_drift_are_refused(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        rotary.base = 20000.0
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, pos)
        rotary.base = 10000.0
        tmp = Path(tempfile.mkdtemp(dir=str(ROOT)))
        self.addCleanup(shutil.rmtree, tmp, True)
        drifted = tmp / "modeling_qwen3_vl.py"
        drifted.write_bytes(HF_COPY.read_bytes() + b"\n# drift\n")
        cache._hf_source_path = drifted
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, pos)

    def test_extra_call_arguments_are_refused_while_enabled(self):
        rotary, cache = make_cache()
        x, pos, cos, sin = prime_like(cache)
        cache.set_enabled(True)
        with self.assertRaises(cache95.TextRopeCacheError):
            rotary.forward(x, pos, "extra")


class Restore(unittest.TestCase):
    def test_uninstall_restores_pristine_and_is_idempotent(self):
        rotary, cache = make_cache()
        self.assertIn("forward", rotary.__dict__)
        first = cache.uninstall()
        self.assertTrue(first["uninstalled"])
        self.assertNotIn("forward", rotary.__dict__)
        self.assertEqual(rotary.forward(hidden(), positions())[0].shape,
                         (1, SEQ, HEAD_DIM))
        second = cache.uninstall()
        self.assertTrue(second["uninstalled"])
        self.assertEqual(len(rotary.original_calls), 1)

    def test_uninstall_refuses_to_clobber_a_third_party_patch(self):
        rotary, cache = make_cache()
        rotary.forward = lambda x, position_ids: ("third-party", None)
        with self.assertRaises(cache95.TextRopeCacheError):
            cache.uninstall()
        self.assertIn("forward", rotary.__dict__)


class SourceAndManifestContract(unittest.TestCase):
    def test_module_scope_is_stdlib_only_and_torch_is_lazy(self):
        tree = ast.parse(MODULE_PATH.read_text())
        module_imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module_imports.add(node.module.split(".")[0])
        self.assertTrue(module_imports <= {"ast", "hashlib", "json", "math", "sys", "time",
                                           "pathlib", "types", "__future__"},
                        module_imports)
        source = MODULE_PATH.read_text()
        # per-instance only: exactly one setattr on the rotary object, no class edit
        self.assertIn("setattr(self._rotary, FORWARD, MethodType(wrapper, self._rotary))",
                      source)
        self.assertNotIn("torch.nn.functional =", source)
        self.assertNotIn("_C._nn.functional", source)
        self.assertNotIn("type(self._rotary).forward", source)
        self.assertNotIn("CPUTextRopeCache95.forward", source)
        self.assertIn("except RuntimeError as error:", source)
        self.assertIn(cache95.INFERENCE_VERSION_ERROR, source)

    def test_manifest_matches_delivered_files_and_stays_non_claiming(self):
        manifest = json.loads((ROOT / "manifest95.json").read_text())
        for name, entry in manifest["files"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            self.assertEqual(entry["sha256"], actual, name)
            self.assertEqual(entry["bytes"], (ROOT / name).stat().st_size, name)
        self.assertEqual(manifest["task"], "task-46")
        self.assertTrue(manifest["source_only"])
        self.assertFalse(manifest["executed_by_author"])
        self.assertFalse(manifest["proves_precision"])
        self.assertFalse(manifest["proves_speed"])
        self.assertFalse(manifest["claims_lossless_native_bits"])
        self.assertIsNone(manifest["numerical_tolerance"])


if __name__ == "__main__":
    unittest.main()