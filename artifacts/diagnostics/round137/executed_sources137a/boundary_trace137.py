"""Return-preserving R137 logical-SHA observations of ORIGINAL module calls.

Select boundaries explicitly, including the native ParametrizationList or its
weight-norm child when actual derived weights are needed. This observer never
accesses a module's dynamic weight getter, invokes a parametrization, installs
parametrize.cached(), or substitutes inputs/outputs. Tensor fingerprints are
computed immediately in the hooks: no detached tensor alias survives a return.
D2H hashing is nonclean observation, not physical-layout or CANN causal proof.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from fingerprints129 import fingerprint, normalized_fingerprint

_REGISTRIES = (
    "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs", "_forward_hooks",
    "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
)


def _scalar_copy(value):
    """Require scalar JSON evidence and copy it, not a tensor/storage alias."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _safe_message(error):
    try:
        return str(error)
    except BaseException:
        return "Exception message unavailable"


class BoundaryTraceCleanupError(RuntimeError):
    def __init__(self, errors):
        self.cleanup_errors = _scalar_copy(errors)
        self.cleanup_unknown = True
        super().__init__("Own boundary hooks are not verified restored: " + json.dumps(errors))


class BoundaryTraceProtocolError(RuntimeError):
    pass


class TraceCaptureError(BoundaryTraceProtocolError):
    """Raised only after the complete original component call has returned."""

    def __init__(self, label, errors, partial_rows, pending):
        self.valid = False
        self.call_label = label
        self.capture_errors = _scalar_copy(errors)
        self.errors = self.capture_errors
        self.partial_rows = _scalar_copy(partial_rows)
        self.pending_boundaries = _scalar_copy(pending)
        super().__init__("Invalid deferred boundary capture: " + label)


class BoundaryTrace137:
    """Install own append-only pre/post hooks on explicit (name, module) pairs.

    API: start_call(label), original module forward(s), end_call() -> scalar rows,
    close() -> cleanup receipt. Positions and invocation ordinals are zero-based.
    POST hooks observe successful original returns. Capture failures are deferred
    until end_call(): pre/post hooks still return None so a peer does not enter a
    TP collective while this rank exits early. Close attempts physical removal
    independently of whether the observed call or its capture completed.
    """

    def __init__(self, torch, modules):
        self.torch = torch
        self._handles = []
        self._owned_callbacks = []
        self._modules = []
        self._active_label = None
        self._rows = []
        self._ordinals = {}
        self._pending = {}
        self._capture_errors = []
        self._invalid_call = False
        self._invalid_trace = False
        self._closed = False
        self.cleanup_errors = []
        self.discarded_active_call = None
        selected = list(modules)
        if not selected:
            raise ValueError("Explicit nonempty boundary selection is required")
        names, identities = set(), set()
        for item in selected:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("Pass explicit (name, module) pairs")
            name, module = item
            if not isinstance(name, str) or not name or name in names or id(module) in identities:
                raise ValueError("Boundary names and module identities must be unique")
            namespace = vars(module)
            if not isinstance(namespace.get("_parameters"), Mapping):
                raise TypeError("Boundary must expose the native direct parameter dictionary")
            # Read only native dictionaries, never computed tensor properties.
            registries = {key: dict(namespace[key]) for key in _REGISTRIES if isinstance(namespace.get(key), Mapping)}
            if "_forward_pre_hooks" not in registries or "_forward_hooks" not in registries:
                raise TypeError("Native hook dictionaries are required to verify exact cleanup")
            self._modules.append({"name": name, "module": module, "original_registries": registries})
            names.add(name)
            identities.add(id(module))
        try:
            for item in self._modules:
                module = item["module"]
                pre, post = self._make_hooks(item)
                self._register(item, "pre", pre, module.register_forward_pre_hook)
                self._register(item, "post", post, module.register_forward_hook)
        except BaseException as primary:
            # Attempt EVERY handle. Do not replace the registration exception with
            # a cleanup failure or claim restoration when an orphan hook remains.
            try:
                errors = self._remove_owned()
            except BaseException as rollback:
                self._closed = False
                errors = [{"operation": "unexpected_rollback_failure", "type": type(rollback).__name__}]
            try:
                primary.boundary_trace_cleanup_errors = _scalar_copy(errors)
                primary.boundary_trace_cleanup_unknown = bool(errors)
            except BaseException:
                pass
            raise

    @property
    def closed(self):
        return self._closed

    def _register(self, item, kind, callback, register):
        # Track even a callback whose registration raises without returning a
        # handle. We never delete an orphan by clearing someone else's registry.
        self._owned_callbacks.append((item, kind, callback))
        handle = register(callback, with_kwargs=True)
        entry = {"item": item, "kind": kind, "handle": handle, "handle_id": None, "removed": False}
        self._handles.append(entry)  # Own it BEFORE reading any handle metadata.
        entry["handle_id"] = getattr(handle, "id", None)

    def _snapshot(self, value):
        return _scalar_copy(fingerprint(value, self.torch, include_pointers=False))

    def _base_row(self, item, ordinal, boundary):
        cls = type(item["module"])
        return {"position": len(self._rows), "name": item["name"], "ordinal": ordinal,
                "boundary": boundary, "call_label": self._active_label,
                "qualified_class": cls.__module__ + "." + cls.__qualname__,
                "logical_SHA_not_physical_layout_or_kernel_proof": True}

    def _mark_capture_error(self, item, boundary, ordinal, failure):
        # Even reporting an exotic exception must not make a hook exit before
        # its peers finish the native component's original collectives.
        self._invalid_call = True
        self._invalid_trace = True
        try:
            self._capture_errors.append({"name": item["name"], "boundary": boundary,
                                         "ordinal": ordinal, "type": type(failure).__name__,
                                         "message": _safe_message(failure)})
        except BaseException:
            pass  # The sticky invalid flag still prevents a successful receipt.

    def _make_hooks(self, item):
        name = item["name"]

        def pre(module, args, kwargs):
            if self._active_label is None:
                return None
            ordinal = None
            try:
                ordinal = self._ordinals.get(name, 0)
                self._ordinals[name] = ordinal + 1
                self._pending.setdefault(name, []).append(ordinal)
                # ONLY the existing direct dictionary. In particular a
                # parametrized Conv may have bias but no direct weight entry;
                # its computed operand must come from the native norm POST.
                parameters = vars(module)["_parameters"]
                row = self._base_row(item, ordinal, "pre")
                row.update(args=self._snapshot(args), kwargs=self._snapshot(kwargs),
                           direct_parameter_names=list(parameters),
                           direct_parameters=self._snapshot(dict(parameters)))
                self._rows.append(row)
            except BaseException as failure:
                self._mark_capture_error(item, "pre", ordinal, failure)
            return None  # Never replace args or abort before native collectives.

        def post(module, args, kwargs, output):
            if self._active_label is None:
                return None
            ordinal = None
            try:
                stack = self._pending.get(name, [])
                if not stack:
                    raise BoundaryTraceProtocolError("POST lacks its PRE bookkeeping: " + name)
                ordinal = stack.pop()
                row = self._base_row(item, ordinal, "post")
                # This immediate fingerprint is crucial for resblock outputs
                # mutated later by xs += output, and for native derived weights.
                row["output"] = self._snapshot(output)
                self._rows.append(row)
            except BaseException as failure:
                self._mark_capture_error(item, "post", ordinal, failure)
            return None  # Never replace output or exit the original call early.

        return pre, post

    def start_call(self, label):
        if self._closed or self.cleanup_errors or self._invalid_trace:
            raise BoundaryTraceProtocolError("Cannot reuse closed/unknown/invalid trace state")
        if self._active_label is not None:
            raise BoundaryTraceProtocolError("A trace call is already active")
        if not isinstance(label, str) or not label:
            raise ValueError("A nonempty diagnostic call label is required")
        self._active_label = label
        self._rows, self._ordinals, self._pending = [], {}, {}
        self._capture_errors = []
        self._invalid_call = False

    def end_call(self):
        if self._active_label is None:
            raise BoundaryTraceProtocolError("No active trace call")
        label, self._active_label = self._active_label, None
        rows = _scalar_copy(self._rows)
        pending = {name: list(items) for name, items in self._pending.items() if items}
        errors = _scalar_copy(self._capture_errors)
        invalid = self._invalid_call
        self._rows, self._ordinals, self._pending = [], {}, {}
        if invalid or pending or not rows:
            self._invalid_trace = True
            if pending or not rows:
                errors.append({"operation": "incomplete_original_call_or_empty_capture", "pending": pending})
            raise TraceCaptureError(label, errors, rows, pending)
        for row in rows:
            row["validated_end_call"] = True
        return rows

    def _remove_owned(self):
        self._closed = False  # Unknown until every removal AND verification finishes.
        errors = []
        for entry in reversed(self._handles):
            if entry["removed"]:
                continue
            try:
                entry["handle"].remove()
                entry["removed"] = True
            except BaseException as failure:
                errors.append({"operation": "remove_own_handle", "name": entry["item"]["name"],
                               "boundary": entry["kind"], "type": type(failure).__name__,
                               "message": _safe_message(failure)})
        for item in self._modules:
            namespace = vars(item["module"])
            for registry, original in item["original_registries"].items():
                current = namespace.get(registry)
                if not isinstance(current, Mapping) or any(key not in current or current[key] is not value for key, value in original.items()):
                    errors.append({"operation": "verify_preexisting_hooks", "name": item["name"], "registry": registry})
        for item, kind, callback in self._owned_callbacks:
            registry = "_forward_pre_hooks" if kind == "pre" else "_forward_hooks"
            current = vars(item["module"]).get(registry)
            if not isinstance(current, Mapping) or any(value is callback for value in current.values()):
                errors.append({"operation": "verify_no_own_or_orphan_hook", "name": item["name"], "boundary": kind})
        for entry in self._handles:
            handle_id = entry["handle_id"]
            if handle_id is None:
                errors.append({"operation": "unknown_handle_identity", "name": entry["item"]["name"], "boundary": entry["kind"]})
                continue
            namespace = vars(entry["item"]["module"])
            if any(isinstance(namespace.get(registry), Mapping) and handle_id in namespace[registry] for registry in _REGISTRIES):
                errors.append({"operation": "verify_no_own_registration_metadata", "name": entry["item"]["name"], "boundary": entry["kind"]})
        self.cleanup_errors = _scalar_copy(errors)
        self._closed = not errors
        return errors

    def close(self):
        if self._active_label is not None:
            self.discarded_active_call = {"label": self._active_label, "completed": False}
            self._active_label = None
            self._rows, self._ordinals, self._pending = [], {}, {}
        errors = self._remove_owned()
        if errors:
            raise BoundaryTraceCleanupError(errors)
        return {"own_hooks_removed": True, "preexisting_hooks_preserved": True,
                "discarded_active_call": self.discarded_active_call,
                "invalid_capture_ever_observed": self._invalid_trace,
                "logical_trace_not_numeric_acceptance_or_CANN_proof": True}


def _sha_paths(value, path=""):
    result = {}
    if isinstance(value, dict):
        if value.get("kind") == "tensor" and "sha256_logical_words" in value:
            result[path or "/"] = value["sha256_logical_words"]
        else:
            for key, item in value.items():
                result.update(_sha_paths(item, path + "/" + str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(_sha_paths(item, path + "/" + str(index)))
    return result


def _signature(row):
    return normalized_fingerprint({"inputs": {"args": row.get("args"), "kwargs": row.get("kwargs")},
                                   "parameters": row.get("direct_parameters"), "output": row.get("output")})


def first_boundary_change(reference_rows, actual_rows):
    """Locate the earliest observed signature/sequence change, not its cause.

    Labels, devices, pointers and version counters do not create a logical-word
    difference. Shape/dtype/stride metadata is retained; different SHA paths are
    separately reported, so a layout-only difference is not called changed data.
    """
    if not isinstance(reference_rows, Sequence) or isinstance(reference_rows, (str, bytes)) or not isinstance(actual_rows, Sequence) or isinstance(actual_rows, (str, bytes)):
        raise TypeError("Compare two sequences of completed scalar trace rows")
    if not reference_rows and not actual_rows:
        raise ValueError("Two absent captures do not establish repeatability")
    if any(not isinstance(row, dict) for row in list(reference_rows) + list(actual_rows)):
        raise TypeError("Boundary rows must be scalar dictionaries")
    if any(row.get("validated_end_call") is not True for row in list(reference_rows) + list(actual_rows)):
        raise ValueError("Invalid/partial captures cannot establish SHA equality")
    for position in range(max(len(reference_rows), len(actual_rows))):
        a = reference_rows[position] if position < len(reference_rows) else None
        b = actual_rows[position] if position < len(actual_rows) else None
        structural_keys = ("name", "ordinal", "boundary", "qualified_class")
        structural = a is None or b is None or any(a.get(key) != b.get(key) for key in structural_keys)
        left, right = _signature(a or {}), _signature(b or {})
        if not structural and left == right:
            continue
        chosen = b if b is not None else a
        result = {"position": position, "name": chosen.get("name"), "ordinal": chosen.get("ordinal"),
                  "boundary": chosen.get("boundary"), "structural_mismatch": structural,
                  "input_changed": left["inputs"] != right["inputs"],
                  "parameter_changed": left["parameters"] != right["parameters"],
                  "output_changed": left["output"] != right["output"],
                  "reference_signature": left, "actual_signature": right,
                  "logical_SHA_only_not_physical_or_operator_cause": True}
        for field, name in (("inputs", "input"), ("parameters", "parameter"), ("output", "output")):
            x, y = _sha_paths(left[field]), _sha_paths(right[field])
            result["different_" + name + "_SHA_paths"] = [
                {"path": path, "reference_SHA": x.get(path), "actual_SHA": y.get(path)}
                for path in sorted(set(x) | set(y)) if x.get(path) != y.get(path)
            ]
        return result
    return None
