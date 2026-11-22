#!/usr/bin/env python3
"""Pure-stdlib symbolic R137 hook units: no Torch, native model or remote work.

The mock module deliberately makes a dynamic weight getter fail. Symbolic native
norm calls are counted, not numerically evaluated. Hashes are JSON byte hashes
of mock values and cannot serve as NPU arithmetic or model-repeatability proof.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import boundary_trace137 as trace_module
from boundary_trace137 import (
    BoundaryTrace137, BoundaryTraceCleanupError, BoundaryTraceProtocolError,
    TraceCaptureError, first_boundary_change,
)


class Tensor:
    def __init__(self, values, *, device="npu:0", dtype="float32"):
        self.values = list(values)
        self.device = device
        self.dtype = dtype
        self.version = 0


def fake_fingerprint(value, torch, *, include_pointers=False):
    if isinstance(value, Tensor):
        return {"kind": "tensor", "shape": [len(value.values)], "dtype": value.dtype,
                "stride": [1], "storage_offset": 0, "logical_bytes": len(value.values) * 4,
                "device": value.device, "tensor_version": value.version,
                "sha256_logical_words": hashlib.sha256(json.dumps(value.values).encode()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): fake_fingerprint(v, torch) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fake_fingerprint(v, torch) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported mock fingerprint value")


class Handle:
    tokens = itertools.count()

    def __init__(self, module, kind, failures=0):
        self.module, self.kind = module, kind
        self.id = next(self.tokens)
        self.remove_calls = 0
        self.failures = failures

    def remove(self):
        self.remove_calls += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("Injected own-handle removal failure")
        names = ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs") if self.kind == "pre" else (
            "_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called"
        )
        for name in names:
            vars(self.module)[name].pop(self.id, None)


class Module:
    def __init__(self, forward=None, parameters=None):
        self._parameters = dict(parameters or {})
        self._forward_pre_hooks = {}
        self._forward_pre_hooks_with_kwargs = {}
        self._forward_hooks = {}
        self._forward_hooks_with_kwargs = {}
        self._forward_hooks_always_called = {}
        self.native_forward = forward or (lambda *args, **kwargs: args[0])
        self.forward_calls = 0
        self.getter_calls = 0
        self.hook_returns = []
        self.handles = []
        self.registration_error = None
        self.remove_failures = {}

    @property
    def weight(self):
        self.getter_calls += 1
        raise AssertionError("Observer evaluated the dynamic weight getter")

    def named_parameters(self, *args, **kwargs):
        raise AssertionError("Observer must not traverse/re-evaluate module parameters")

    def _register(self, callback, kind, with_kwargs):
        error = self.registration_error
        if error is not None and error[0] == kind and error[1] == "before":
            raise error[2]
        handle = Handle(self, kind, self.remove_failures.get(kind, 0))
        self.handles.append(handle)
        hooks = self._forward_pre_hooks if kind == "pre" else self._forward_hooks
        flags = self._forward_pre_hooks_with_kwargs if kind == "pre" else self._forward_hooks_with_kwargs
        hooks[handle.id] = callback
        if with_kwargs:
            flags[handle.id] = True
        if error is not None and error[0] == kind and error[1] == "after":
            raise error[2]
        return handle

    def register_forward_pre_hook(self, callback, *, with_kwargs=False):
        return self._register(callback, "pre", with_kwargs)

    def register_forward_hook(self, callback, *, with_kwargs=False):
        return self._register(callback, "post", with_kwargs)

    def __call__(self, *args, **kwargs):
        for token, callback in list(self._forward_pre_hooks.items()):
            result = callback(self, args, kwargs) if token in self._forward_pre_hooks_with_kwargs else callback(self, args)
            self.hook_returns.append((token, "pre", result))
            if result is not None:
                args, kwargs = result
        self.forward_calls += 1
        output = self.native_forward(*args, **kwargs)
        for token, callback in list(self._forward_hooks.items()):
            result = callback(self, args, kwargs, output) if token in self._forward_hooks_with_kwargs else callback(self, args, output)
            self.hook_returns.append((token, "post", result))
            if result is not None:
                output = result
        return output


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.mock_fingerprint = patch.object(trace_module, "fingerprint", fake_fingerprint)
        self.mock_fingerprint.start()
        self.addCleanup(self.mock_fingerprint.stop)
        self.torch = object()  # No Torch methods exist, let alone get executed.

    def test_direct_parameters_and_nested_kwargs_without_getter(self):
        value, weight = Tensor([1, 2]), Tensor([3, 4])
        module = Module(parameters={"weight": weight, "bias": None})
        observer = BoundaryTrace137(self.torch, [("embedding", module)])
        observer.start_call("fixed0")
        self.assertIs(module(value, optional={"nested": [Tensor([5]), None]}), value)
        rows = observer.end_call()
        self.assertEqual([r["boundary"] for r in rows], ["pre", "post"])
        self.assertEqual(rows[0]["direct_parameter_names"], ["weight", "bias"])
        self.assertEqual(rows[0]["direct_parameters"]["weight"]["sha256_logical_words"], fake_fingerprint(weight, self.torch)["sha256_logical_words"])
        self.assertIsNone(rows[0]["direct_parameters"]["bias"])
        self.assertEqual(module.getter_calls, 0)
        self.assertTrue(all(result is None for _, _, result in module.hook_returns))
        self.assertTrue(observer.close()["own_hooks_removed"])
        self.assertTrue(observer.closed)

    def test_native_parametrization_list_and_child_compute_once(self):
        original0, original1 = Tensor([1]), Tensor([2, 3])
        derived = Tensor([99, 100])
        child = Module(forward=lambda g, v: derived)
        parameters = Module(forward=lambda: child(original0, original1), parameters={"original0": original0, "original1": original1})
        output, consumed = Tensor([7]), []
        def native_conv(x):
            consumed.append(parameters())  # Existing symbolic native operation.
            return output
        conv = Module(forward=native_conv, parameters={"bias": None})
        observer = BoundaryTrace137(self.torch, [("conv", conv), ("conv.parametrization", parameters), ("conv.native_weight_norm", child)])
        observer.start_call("audio0")
        self.assertIs(conv(Tensor([8])), output)
        rows = observer.end_call()
        self.assertEqual([r["name"] for r in rows], ["conv", "conv.parametrization", "conv.native_weight_norm", "conv.native_weight_norm", "conv.parametrization", "conv"])
        self.assertEqual(parameters.forward_calls, 1)
        self.assertEqual(child.forward_calls, 1)
        self.assertEqual(conv.forward_calls, 1)
        self.assertIs(consumed[0], derived)
        self.assertEqual(rows[1]["direct_parameter_names"], ["original0", "original1"])
        self.assertEqual(rows[2]["args"][0]["sha256_logical_words"], fake_fingerprint(original0, self.torch)["sha256_logical_words"])
        expected = fake_fingerprint(derived, self.torch)["sha256_logical_words"]
        self.assertEqual(rows[3]["output"]["sha256_logical_words"], expected)
        self.assertEqual(rows[4]["output"]["sha256_logical_words"], expected)
        self.assertEqual(sum(m.getter_calls for m in (conv, parameters, child)), 0)
        self.assertTrue(all(result is None for m in (conv, parameters, child) for _, _, result in m.hook_returns))
        observer.close()

    def test_immediate_output_and_input_hash_before_later_inplace_mutation(self):
        value, parameter = Tensor([1]), Tensor([2])
        module = Module(parameters={"original0": parameter})
        observer = BoundaryTrace137(self.torch, [("first_resblock", module)])
        before = fake_fingerprint(value, self.torch)["sha256_logical_words"]
        old_parameter = fake_fingerprint(parameter, self.torch)["sha256_logical_words"]
        observer.start_call("audio")
        result = module(value)
        result.values += [999]  # Simulates the caller's later xs += output.
        parameter.values[0] = 8
        rows = observer.end_call()
        self.assertEqual(rows[0]["args"][0]["sha256_logical_words"], before)
        self.assertEqual(rows[1]["output"]["sha256_logical_words"], before)
        self.assertEqual(rows[0]["direct_parameters"]["original0"]["sha256_logical_words"], old_parameter)
        self.assertNotEqual(rows[1]["output"]["sha256_logical_words"], fake_fingerprint(result, self.torch)["sha256_logical_words"])
        json.dumps(rows, allow_nan=False)
        observer.close()

    def test_reentrant_module_ordinals_follow_actual_occurrences(self):
        value = Tensor([1])
        module = Module()
        def forward(x, depth=0):
            return module(x, depth=depth-1) if depth else x
        module.native_forward = forward
        observer = BoundaryTrace137(self.torch, [("shared", module)])
        observer.start_call("nested")
        self.assertIs(module(value, depth=1), value)
        rows = observer.end_call()
        self.assertEqual([(r["boundary"], r["ordinal"]) for r in rows], [("pre", 0), ("pre", 1), ("post", 1), ("post", 0)])
        self.assertEqual([r["position"] for r in rows], list(range(4)))
        observer.close()

    def test_existing_hooks_and_their_output_replacement_preserved(self):
        value, replacement = Tensor([1]), Tensor([42])
        module = Module()
        pre = module.register_forward_pre_hook(lambda m, a, k: None, with_kwargs=True)
        post = module.register_forward_hook(lambda m, a, k, o: replacement, with_kwargs=True)
        original_pre, original_post = dict(module._forward_pre_hooks), dict(module._forward_hooks)
        observer = BoundaryTrace137(self.torch, [("native", module)])
        observer.start_call("observed")
        self.assertIs(module(value), replacement)
        rows = observer.end_call()
        self.assertEqual(rows[1]["output"]["sha256_logical_words"], fake_fingerprint(replacement, self.torch)["sha256_logical_words"])
        observer.close()
        self.assertEqual(module._forward_pre_hooks, original_pre)
        self.assertEqual(module._forward_hooks, original_post)
        self.assertEqual(pre.remove_calls + post.remove_calls, 0)
        observer.close()  # Idempotent, no second removals of old or own hooks.

    def test_registration_failure_rolls_back_every_returned_handle(self):
        first, second = Module(), Module()
        primary = RuntimeError("Register POST failed")
        second.registration_error = ("post", "before", primary)
        with self.assertRaises(RuntimeError) as caught:
            BoundaryTrace137(self.torch, [("first", first), ("second", second)])
        self.assertIs(caught.exception, primary)
        self.assertFalse(primary.boundary_trace_cleanup_unknown)
        self.assertEqual(primary.boundary_trace_cleanup_errors, [])
        self.assertEqual([h.remove_calls for m in (first, second) for h in m.handles], [1, 1, 1])
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in (first, second)))

    def test_constructor_primary_preserved_when_one_removal_fails(self):
        first, second = Module(), Module()
        first.remove_failures["post"] = 1
        primary = RuntimeError("Registration failure is primary")
        second.registration_error = ("post", "before", primary)
        with self.assertRaises(RuntimeError) as caught:
            BoundaryTrace137(self.torch, [("first", first), ("second", second)])
        self.assertIs(caught.exception, primary)
        self.assertTrue(primary.boundary_trace_cleanup_unknown)
        self.assertTrue(primary.boundary_trace_cleanup_errors)
        self.assertEqual([h.remove_calls for m in (first, second) for h in m.handles], [1, 1, 1])
        self.assertFalse(first._forward_pre_hooks)
        self.assertTrue(first._forward_hooks)

    def test_registration_that_mutates_then_raises_is_unknown_not_false_restore(self):
        module = Module()
        primary = RuntimeError("No POST handle returned")
        module.registration_error = ("post", "after", primary)
        with self.assertRaises(RuntimeError) as caught:
            BoundaryTrace137(self.torch, [("orphan", module)])
        self.assertIs(caught.exception, primary)
        self.assertTrue(primary.boundary_trace_cleanup_unknown)
        self.assertFalse(module._forward_pre_hooks)
        self.assertTrue(module._forward_hooks)  # No clearing unowned/unknown IDs.
        self.assertEqual([h.remove_calls for h in module.handles], [1, 0])

    def test_close_removes_all_handles_independently_and_allows_cleanup_retry(self):
        module = Module()
        module.remove_failures["post"] = 1
        observer = BoundaryTrace137(self.torch, [("conv", module)])
        with self.assertRaises(BoundaryTraceCleanupError) as caught:
            observer.close()
        self.assertTrue(caught.exception.cleanup_unknown)
        self.assertFalse(observer.closed)
        self.assertEqual([h.remove_calls for h in module.handles], [1, 1])
        self.assertFalse(module._forward_pre_hooks)
        self.assertTrue(module._forward_hooks)
        with self.assertRaises(BoundaryTraceProtocolError):
            observer.start_call("invalid")
        self.assertTrue(observer.close()["own_hooks_removed"])
        self.assertTrue(observer.closed)
        self.assertFalse(module._forward_hooks)

    def test_unexpected_existing_hook_mutation_not_overwritten(self):
        module = Module()
        old = module.register_forward_hook(lambda m, a, k, o: None, with_kwargs=True)
        observer = BoundaryTrace137(self.torch, [("conv", module)])
        unrelated = lambda m, a, k, o: None
        module._forward_hooks[old.id] = unrelated
        with self.assertRaises(BoundaryTraceCleanupError):
            observer.close()
        self.assertIs(module._forward_hooks[old.id], unrelated)
        self.assertEqual(old.remove_calls, 0)
        self.assertFalse(observer.closed)

    def test_pre_or_post_capture_error_is_deferred_until_native_return(self):
        for fail_at in (1, 2, 3, 4):  # PRE args/kwargs/params or POST output.
            with self.subTest(fail_at=fail_at):
                value, output = Tensor([1]), Tensor([2])
                module = Module(forward=lambda x: output)
                observer = BoundaryTrace137(self.torch, [("collective_component", module)])
                calls = 0
                def failing(value, torch, *, include_pointers=False):
                    nonlocal calls
                    calls += 1
                    if calls == fail_at:
                        raise RuntimeError("Injected SHA capture failure")
                    return fake_fingerprint(value, torch)
                with patch.object(trace_module, "fingerprint", failing):
                    observer.start_call("fixed")
                    self.assertIs(module(value), output)  # Original call still finishes.
                    self.assertEqual(module.forward_calls, 1)
                    self.assertTrue(all(result is None for _, _, result in module.hook_returns))
                    with self.assertRaises(TraceCaptureError) as caught:
                        observer.end_call()
                self.assertFalse(caught.exception.valid)
                self.assertEqual(caught.exception.capture_errors[0]["message"], "Injected SHA capture failure")
                with self.assertRaises(BoundaryTraceProtocolError):
                    observer.start_call("not_allowed_to_repeat_invalid_capture")
                observer.close()

    def test_exception_message_failure_still_deferred_and_invalid(self):
        class Unprintable(RuntimeError):
            def __str__(self):raise ValueError("No printable exception")
        module = Module()
        observer = BoundaryTrace137(self.torch, [("native", module)])
        value = Tensor([1])
        with patch.object(trace_module, "fingerprint", side_effect=Unprintable()):
            observer.start_call("fixed")
            self.assertIs(module(value), value)
            with self.assertRaises(TraceCaptureError) as caught:
                observer.end_call()
        self.assertFalse(caught.exception.valid)
        observer.close()

    def test_original_forward_exception_not_replaced_by_observation(self):
        primary = RuntimeError("Native failure")
        def native(x):raise primary
        module = Module(forward=native)
        observer = BoundaryTrace137(self.torch, [("broken", module)])
        observer.start_call("bad")
        with self.assertRaises(RuntimeError) as caught:
            module(Tensor([1]))
        self.assertIs(caught.exception, primary)
        with self.assertRaises(TraceCaptureError) as incomplete:
            observer.end_call()
        self.assertEqual(incomplete.exception.pending_boundaries, {"broken": [0]})
        observer.close()

    def test_close_during_aborted_call_does_not_claim_completed_capture(self):
        module = Module()
        observer = BoundaryTrace137(self.torch, [("native", module)])
        observer.start_call("aborted")
        receipt = observer.close()
        self.assertTrue(receipt["own_hooks_removed"])
        self.assertEqual(receipt["discarded_active_call"], {"label": "aborted", "completed": False})
        with self.assertRaises(BoundaryTraceProtocolError):
            observer.end_call()

    def test_protocol_and_duplicate_selection_gates(self):
        module = Module()
        with self.assertRaises(ValueError):BoundaryTrace137(self.torch, [])
        with self.assertRaises(ValueError):BoundaryTrace137(self.torch, [("a", module), ("b", module)])
        observer = BoundaryTrace137(self.torch, [("a", module)])
        with self.assertRaises(BoundaryTraceProtocolError):observer.end_call()
        observer.start_call("active")
        with self.assertRaises(BoundaryTraceProtocolError):observer.start_call("second")
        with self.assertRaises(TraceCaptureError):observer.end_call()  # No observed forward.
        observer.close()

    def completed_rows(self, module, label):
        observer = BoundaryTrace137(self.torch, [("native", module)])
        observer.start_call(label)
        module(Tensor([1]))
        rows = observer.end_call()
        observer.close()
        return rows

    def test_first_change_ignores_call_labels_device_and_version_not_SHA(self):
        module = Module()
        first = self.completed_rows(module, "original0")
        second = copy.deepcopy(first)
        for row in second:
            row["call_label"] = "original1"
        second[0]["args"][0]["device"] = "npu:7"
        second[0]["args"][0]["tensor_version"] = 99
        self.assertIsNone(first_boundary_change(first, second))
        second[1]["output"]["sha256_logical_words"] = "f" * 64
        change = first_boundary_change(first, second)
        self.assertEqual((change["position"], change["name"], change["ordinal"], change["boundary"]), (1, "native", 0, "post"))
        self.assertFalse(change["input_changed"])
        self.assertFalse(change["parameter_changed"])
        self.assertTrue(change["output_changed"])
        self.assertEqual(len(change["different_output_SHA_paths"]), 1)
        self.assertTrue(change["logical_SHA_only_not_physical_or_operator_cause"])

    def test_first_input_parameter_layout_and_structural_changes_separate(self):
        module = Module(parameters={"original0": Tensor([2])})
        first = self.completed_rows(module, "a")
        for field in ("input", "parameter", "layout", "structure"):
            second = copy.deepcopy(first)
            if field == "input":second[0]["args"][0]["sha256_logical_words"] = "a" * 64
            elif field == "parameter":second[0]["direct_parameters"]["original0"]["sha256_logical_words"] = "b" * 64
            elif field == "layout":second[0]["args"][0]["stride"] = [2]
            else:second[0]["name"] = "other"
            change = first_boundary_change(first, second)
            self.assertEqual(change["position"], 0)
            if field == "layout":
                self.assertTrue(change["input_changed"])
                self.assertEqual(change["different_input_SHA_paths"], [])
            elif field == "structure":self.assertTrue(change["structural_mismatch"])
            else:self.assertTrue(change[field + "_changed"])
        change = first_boundary_change(first, first[:1])
        self.assertTrue(change["structural_mismatch"])
        self.assertEqual(change["position"], 1)

    def test_returned_handle_owned_before_failing_handle_metadata_read(self):
        module = Module()
        primary = RuntimeError("Handle identity accessor failed")
        registered = module.register_forward_pre_hook
        class BadIdentity:
            @property
            def id(self):raise primary
            def remove(self):delegate.remove()
        delegate = None
        def registration(callback, *, with_kwargs):
            nonlocal delegate
            delegate = registered(callback, with_kwargs=with_kwargs)
            return BadIdentity()
        module.register_forward_pre_hook = registration
        with self.assertRaises(RuntimeError) as caught:
            BoundaryTrace137(self.torch, [("native", module)])
        self.assertIs(caught.exception, primary)
        self.assertEqual(delegate.remove_calls, 1)
        self.assertFalse(module._forward_pre_hooks)
        self.assertTrue(primary.boundary_trace_cleanup_unknown)  # ID metadata unknown.

    def test_unprintable_cleanup_error_does_not_skip_other_handle_removal(self):
        class Unprintable(RuntimeError):
            def __str__(self):raise ValueError("No printable exception")
        module = Module()
        observer = BoundaryTrace137(self.torch, [("native", module)])
        post = module.handles[1]
        def failure():
            post.remove_calls += 1
            raise Unprintable()
        post.remove = failure
        with self.assertRaises(BoundaryTraceCleanupError) as caught:
            observer.close()
        self.assertEqual([h.remove_calls for h in module.handles], [1, 1])
        self.assertEqual(caught.exception.cleanup_errors[0]["message"], "Exception message unavailable")
        self.assertFalse(observer.closed)
        self.assertFalse(module._forward_pre_hooks)

    def test_nonfinite_scalar_metadata_capture_is_invalid_not_native_abort(self):
        module = Module()
        value = Tensor([1])
        observer = BoundaryTrace137(self.torch, [("native", module)])
        observer.start_call("metadata")
        self.assertIs(module(value, metadata={"invalid_scalar": float("nan")}), value)
        with self.assertRaises(TraceCaptureError):
            observer.end_call()
        self.assertTrue(observer.close()["invalid_capture_ever_observed"])

    def test_absent_or_malformed_comparison_is_not_false_repeatability(self):
        with self.assertRaises(ValueError):first_boundary_change([], [])
        with self.assertRaises(TypeError):first_boundary_change("not rows", [])
        with self.assertRaises(TypeError):first_boundary_change(["not row dict"], [])

    def test_partial_capture_not_compared_as_equal_even_when_same_error_repeats(self):
        module = Module()
        observer = BoundaryTrace137(self.torch, [("native", module)])
        value = Tensor([1])
        calls = 0
        def fail_post(value, torch, *, include_pointers=False):
            nonlocal calls
            calls += 1
            if calls == 4:raise RuntimeError("POST SHA failed")
            return fake_fingerprint(value, torch)
        with patch.object(trace_module, "fingerprint", fail_post):
            observer.start_call("invalid")
            self.assertIs(module(value), value)
            with self.assertRaises(TraceCaptureError) as caught:
                observer.end_call()
        partial = caught.exception.partial_rows
        self.assertTrue(partial)
        with self.assertRaises(ValueError):first_boundary_change(partial, partial)
        observer.close()

    def test_source_no_Torch_import_getter_or_cached_parametrization_call(self):
        path = Path(__file__).resolve().parent / "boundary_trace137.py"
        tree = ast.parse(path.read_text())
        imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertNotIn("torch", imports)
        self.assertNotIn("numpy", imports)
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "weight" for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" and len(n.args) > 1 and isinstance(n.args[1], ast.Constant) and n.args[1].value == "weight" for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "cached" for n in ast.walk(tree)))


if __name__ == "__main__":
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    unittest.main()
