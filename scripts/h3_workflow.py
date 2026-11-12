#!/usr/bin/env python3
"""Workflow editing and UI-to-API conversion shared by the H3 harness."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


class WorkflowError(ValueError):
    """Raised when a workflow cannot be edited safely."""


@dataclass
class Overrides:
    model: str | None = None
    clip: str | None = None
    video_vae: str | None = None
    audio_vae: str | None = None
    lora: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    steps: int | None = None
    npu_count: int | None = None
    prompt: str | None = None
    seed: int | None = None
    first_image: str | None = None
    last_image: str | None = None
    filename_prefix: str | None = None

    def validate(self) -> None:
        if self.width is not None and (self.width <= 0 or self.width % 32):
            raise WorkflowError("width must be a positive multiple of 32")
        if self.height is not None and (self.height <= 0 or self.height % 32):
            raise WorkflowError("height must be a positive multiple of 32")
        if self.frames is not None and (self.frames < 5 or self.frames % 17 != 5):
            raise WorkflowError(
                "frames must be >=5 and follow the H3 17k+5 alignment "
                "(examples: 124, 192, 345)"
            )
        if self.steps is not None and self.steps <= 0:
            raise WorkflowError("steps must be positive")
        if self.npu_count is not None and self.npu_count not in {1, 2, 4}:
            raise WorkflowError(
                "npu_count must be one of 1, 2, 4 for the ComfyUI H3 patch"
            )
        if self.seed is not None and self.seed < 0:
            raise WorkflowError("seed must be non-negative")


def load_workflow(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read workflow {path}: {exc}") from exc
    workflow, _ = normalize_workflow(data)
    return workflow


def normalize_workflow(data: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(data, dict):
        raise WorkflowError("workflow root must be a JSON object")
    if isinstance(data.get("nodes"), list):
        return data, "ui"
    wrapped = data.get("prompt")
    if isinstance(wrapped, dict) and _looks_like_api(wrapped):
        return wrapped, "api"
    if _looks_like_api(data):
        return data, "api"
    raise WorkflowError("unrecognized workflow: expected ComfyUI UI or API format")


def detect_format(workflow: dict[str, Any]) -> str:
    return "ui" if isinstance(workflow.get("nodes"), list) else "api"


def _looks_like_api(value: dict[str, Any]) -> bool:
    return bool(value) and all(
        isinstance(node, dict) and isinstance(node.get("class_type"), str)
        for node in value.values()
    )


def dump_workflow(workflow: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _nodes(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if detect_format(workflow) == "ui":
        return {str(node["id"]): node for node in workflow["nodes"]}
    return {str(node_id): node for node_id, node in workflow.items()}


def _node_type(node: dict[str, Any]) -> str:
    return str(node.get("type") or node.get("class_type") or "")


def _find_by_type(
    workflow: dict[str, Any], node_type: str, *, required: bool = True
) -> list[dict[str, Any]]:
    matches = [
        node for node in _nodes(workflow).values() if _node_type(node) == node_type
    ]
    if required and not matches:
        raise WorkflowError(f"workflow has no {node_type} node")
    return matches


def _one_by_type(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = _find_by_type(workflow, node_type)
    if len(matches) != 1:
        raise WorkflowError(
            f"expected one {node_type} node, found {len(matches)}"
        )
    return matches[0]


def _ui_input(node: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in node.get("inputs", []):
        if candidate.get("name") == name:
            return candidate
    raise WorkflowError(
        f"{_node_type(node)} node {node.get('id')} has no input {name!r}"
    )


def _set_ui_widget(node: dict[str, Any], name: str, value: Any) -> None:
    widget_names = [
        item["name"]
        for item in node.get("inputs", [])
        if isinstance(item.get("widget"), dict) and item.get("name")
    ]
    if name not in widget_names:
        raise WorkflowError(
            f"{_node_type(node)} node {node.get('id')} has no widget {name!r}"
        )
    values = node.setdefault("widgets_values", [])
    if isinstance(values, dict):
        values[name] = value
        return
    if not isinstance(values, list):
        raise WorkflowError(
            f"{_node_type(node)} node {node.get('id')} has invalid widgets_values"
        )
    index = widget_names.index(name)
    while len(values) <= index:
        values.append(None)
    values[index] = value


def _ui_link(workflow: dict[str, Any], link_id: Any) -> list[Any] | None:
    links = workflow.get("links", [])
    if not isinstance(links, list):
        raise WorkflowError("UI workflow links must be a list")
    for link in links:
        if isinstance(link, list) and len(link) >= 5 and link[0] == link_id:
            return link
    return None


def _disconnect_ui_input(
    workflow: dict[str, Any], node: dict[str, Any], name: str
) -> None:
    ui_input = _ui_input(node, name)
    link_id = ui_input.get("link")
    if link_id is None:
        return
    link = _ui_link(workflow, link_id)
    if link is not None:
        source_id = link[1]
        source = _nodes(workflow).get(str(source_id))
        if source is not None:
            for output in source.get("outputs", []):
                output_links = output.get("links")
                if isinstance(output_links, list):
                    output["links"] = [
                        item for item in output_links if item != link_id
                    ]
        workflow["links"] = [
            item
            for item in workflow.get("links", [])
            if not (isinstance(item, list) and item and item[0] == link_id)
        ]
    ui_input["link"] = None


def _source_for_ui_input(
    workflow: dict[str, Any], node: dict[str, Any], name: str
) -> dict[str, Any] | None:
    ui_input = _ui_input(node, name)
    link_id = ui_input.get("link")
    if link_id is None:
        return None
    link = _ui_link(workflow, link_id)
    if link is None:
        raise WorkflowError(
            f"dangling link {link_id!r} into {_node_type(node)} input {name!r}"
        )
    source = _nodes(workflow).get(str(link[1]))
    if source is None:
        raise WorkflowError(f"link {link_id!r} has missing source node {link[1]!r}")
    return source


def _vae_roles(workflow: dict[str, Any]) -> dict[str, Any]:
    fmt = detect_format(workflow)
    nodes = _nodes(workflow)
    roles: dict[str, Any] = {}
    if fmt == "ui":
        for link in workflow.get("links", []):
            if not isinstance(link, list) or len(link) < 5:
                continue
            _, source_id, _source_slot, target_id, target_slot = link[:5]
            target = nodes.get(str(target_id))
            if target is None or _node_type(target) != "MultiNPUParallelConfig":
                continue
            inputs = target.get("inputs", [])
            if not isinstance(target_slot, int) or target_slot >= len(inputs):
                continue
            role = inputs[target_slot].get("name")
            source = nodes.get(str(source_id))
            if role in {"video_vae", "audio_vae"} and source is not None:
                roles[role] = source
        return roles

    for node in nodes.values():
        if _node_type(node) != "MultiNPUParallelConfig":
            continue
        inputs = node.get("inputs", {})
        for role in ("video_vae", "audio_vae"):
            upstream = inputs.get(role)
            if (
                isinstance(upstream, list)
                and len(upstream) >= 1
                and str(upstream[0]) in nodes
            ):
                roles[role] = nodes[str(upstream[0])]
    return roles


def _set_api_input(node: dict[str, Any], name: str, value: Any) -> None:
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        raise WorkflowError(
            f"{_node_type(node)} node has invalid API inputs object"
        )
    inputs[name] = value


def _api_upstream(
    workflow: dict[str, Any], value: Any
) -> dict[str, Any] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    node = _nodes(workflow).get(str(value[0]))
    if node is None:
        raise WorkflowError(f"API input references missing node {value[0]!r}")
    return node


def _apply_ui_overrides(
    workflow: dict[str, Any], overrides: Overrides
) -> dict[str, Any]:
    if overrides.model:
        _set_ui_widget(_one_by_type(workflow, "UNETLoader"), "unet_name", overrides.model)
    if overrides.clip:
        _set_ui_widget(_one_by_type(workflow, "CLIPLoader"), "clip_name", overrides.clip)
    if overrides.lora:
        _set_ui_widget(
            _one_by_type(workflow, "MiniMaxH3TurboLoRA"),
            "lora_name",
            overrides.lora,
        )

    roles = _vae_roles(workflow)
    if overrides.video_vae:
        source = roles.get("video_vae")
        if source is None or _node_type(source) != "VAELoader":
            raise WorkflowError(
                "cannot identify the video VAE loader from MultiNPUParallelConfig.video_vae"
            )
        _set_ui_widget(source, "vae_name", overrides.video_vae)
    if overrides.audio_vae:
        source = roles.get("audio_vae")
        if source is None or _node_type(source) != "VAELoader":
            raise WorkflowError(
                "cannot identify the audio VAE loader from MultiNPUParallelConfig.audio_vae"
            )
        _set_ui_widget(source, "vae_name", overrides.audio_vae)

    video = _one_by_type(workflow, "MiniMaxH3ImageToVideo")
    if overrides.prompt is not None:
        _disconnect_ui_input(workflow, video, "prompt")
        _set_ui_widget(video, "prompt", overrides.prompt)
    for name, value in (("width", overrides.width), ("height", overrides.height)):
        if value is not None:
            _disconnect_ui_input(workflow, video, name)
            _set_ui_widget(video, name, value)
    if overrides.frames is not None:
        _disconnect_ui_input(workflow, video, "length")
        _set_ui_widget(video, "length", overrides.frames)

    if overrides.steps is not None:
        _set_ui_widget(
            _one_by_type(workflow, "BasicScheduler"), "steps", overrides.steps
        )
    if overrides.seed is not None:
        _set_ui_widget(
            _one_by_type(workflow, "RandomNoise"), "noise_seed", overrides.seed
        )
    if overrides.npu_count is not None:
        _set_ui_widget(
            _one_by_type(workflow, "MultiNPUParallelConfig"),
            "npu_count",
            str(overrides.npu_count),
        )
    if overrides.filename_prefix:
        _set_ui_widget(
            _one_by_type(workflow, "SaveVideo"),
            "filename_prefix",
            overrides.filename_prefix,
        )

    for input_name, image_name in (
        ("first_frame", overrides.first_image),
        ("last_frame", overrides.last_image),
    ):
        if image_name is None:
            continue
        if image_name.lower() in {"none", "null", "-"}:
            _disconnect_ui_input(workflow, video, input_name)
            continue
        source = _source_for_ui_input(workflow, video, input_name)
        if source is None or _node_type(source) != "LoadImage":
            raise WorkflowError(
                f"{input_name} is not connected to a LoadImage node; "
                "use 'none' to remove the image input"
            )
        _set_ui_widget(source, "image", image_name)

    return workflow


def _apply_api_overrides(
    workflow: dict[str, Any], overrides: Overrides
) -> dict[str, Any]:
    if overrides.model:
        _set_api_input(
            _one_by_type(workflow, "UNETLoader"), "unet_name", overrides.model
        )
    if overrides.clip:
        _set_api_input(
            _one_by_type(workflow, "CLIPLoader"), "clip_name", overrides.clip
        )
    if overrides.lora:
        _set_api_input(
            _one_by_type(workflow, "MiniMaxH3TurboLoRA"),
            "lora_name",
            overrides.lora,
        )

    roles = _vae_roles(workflow)
    if overrides.video_vae:
        source = roles.get("video_vae")
        if source is None or _node_type(source) != "VAELoader":
            raise WorkflowError(
                "cannot identify the video VAE loader from MultiNPUParallelConfig.video_vae"
            )
        _set_api_input(source, "vae_name", overrides.video_vae)
    if overrides.audio_vae:
        source = roles.get("audio_vae")
        if source is None or _node_type(source) != "VAELoader":
            raise WorkflowError(
                "cannot identify the audio VAE loader from MultiNPUParallelConfig.audio_vae"
            )
        _set_api_input(source, "vae_name", overrides.audio_vae)

    video = _one_by_type(workflow, "MiniMaxH3ImageToVideo")
    if overrides.prompt is not None:
        _set_api_input(video, "prompt", overrides.prompt)
    for name, value in (
        ("width", overrides.width),
        ("height", overrides.height),
        ("length", overrides.frames),
    ):
        if value is not None:
            _set_api_input(video, name, value)

    if overrides.steps is not None:
        _set_api_input(
            _one_by_type(workflow, "BasicScheduler"), "steps", overrides.steps
        )
    if overrides.seed is not None:
        _set_api_input(
            _one_by_type(workflow, "RandomNoise"), "noise_seed", overrides.seed
        )
    if overrides.npu_count is not None:
        _set_api_input(
            _one_by_type(workflow, "MultiNPUParallelConfig"),
            "npu_count",
            str(overrides.npu_count),
        )
    if overrides.filename_prefix:
        _set_api_input(
            _one_by_type(workflow, "SaveVideo"),
            "filename_prefix",
            overrides.filename_prefix,
        )

    for input_name, image_name in (
        ("first_frame", overrides.first_image),
        ("last_frame", overrides.last_image),
    ):
        if image_name is None:
            continue
        if image_name.lower() in {"none", "null", "-"}:
            video.setdefault("inputs", {}).pop(input_name, None)
            continue
        source = _api_upstream(workflow, video.get("inputs", {}).get(input_name))
        if source is None or _node_type(source) != "LoadImage":
            raise WorkflowError(
                f"{input_name} is not connected to a LoadImage node; "
                "use 'none' to remove the image input"
            )
        _set_api_input(source, "image", image_name)

    return workflow


def apply_overrides(
    workflow: dict[str, Any], overrides: Overrides
) -> tuple[dict[str, Any], dict[str, Any]]:
    overrides.validate()
    modified = copy.deepcopy(workflow)
    fmt = detect_format(modified)
    if fmt == "ui":
        modified = _apply_ui_overrides(modified, overrides)
    else:
        modified = _apply_api_overrides(modified, overrides)
    validate_workflow(modified)
    summary = {
        "format": fmt,
        "video_node": str(_one_by_type(modified, "MiniMaxH3ImageToVideo")["id"])
        if fmt == "ui"
        else next(
            node_id
            for node_id, node in modified.items()
            if _node_type(node) == "MiniMaxH3ImageToVideo"
        ),
        "overrides": {key: value for key, value in asdict(overrides).items() if value is not None},
    }
    return modified, summary


def _spec_inputs(
    object_info: dict[str, Any], node_type: str
) -> list[tuple[str, Any]]:
    if node_type not in object_info:
        raise WorkflowError(f"ComfyUI /object_info has no node type {node_type}")
    node_info = object_info[node_type]
    spec = node_info.get("input") or {}
    ordered: list[tuple[str, Any]] = []
    for section in ("required", "optional"):
        items = spec.get(section) or {}
        if not isinstance(items, dict):
            raise WorkflowError(
                f"invalid {node_type}.input.{section} in /object_info"
            )
        for name, definition in items.items():
            if name not in {item[0] for item in ordered}:
                ordered.append((name, definition))
    return ordered


def _widget_values_for_ui_node(node: dict[str, Any]) -> dict[str, Any]:
    values = node.get("widgets_values", [])
    widget_names = [
        item["name"]
        for item in node.get("inputs", [])
        if isinstance(item.get("widget"), dict) and item.get("name")
    ]
    if isinstance(values, dict):
        return {name: values[name] for name in widget_names if name in values}
    if not isinstance(values, list):
        return {}

    mapped: dict[str, Any] = {}
    position = 0
    for name in widget_names:
        if position >= len(values):
            break
        mapped[name] = values[position]
        position += 1
        if name in {"seed", "noise_seed"} and position < len(values):
            control = values[position]
            if control in {"fixed", "increment", "decrement", "randomize"}:
                position += 1
    return mapped


def ui_to_api(
    workflow: dict[str, Any], object_info: dict[str, Any]
) -> dict[str, Any]:
    if detect_format(workflow) != "ui":
        return copy.deepcopy(workflow)
    nodes = _nodes(workflow)
    links: dict[Any, tuple[Any, Any]] = {}
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 3:
            raise WorkflowError(f"invalid UI link: {link!r}")
        links[link[0]] = (link[1], link[2])

    prompt: dict[str, Any] = {}
    for node_id, node in nodes.items():
        node_type = _node_type(node)
        if node_type in {"Note", "MarkdownNote"} or node.get("mode") == 4:
            continue
        ordered = _spec_inputs(object_info, node_type)
        ui_inputs = {
            item["name"]: item
            for item in node.get("inputs", [])
            if item.get("name")
        }
        widgets = _widget_values_for_ui_node(node)
        inputs: dict[str, Any] = {}

        for name, definition in ordered:
            ui_input = ui_inputs.get(name)
            link_id = ui_input.get("link") if ui_input else None
            if link_id is not None:
                if link_id not in links:
                    raise WorkflowError(
                        f"dangling link {link_id!r} in {node_type} node {node_id}"
                    )
                source_id, source_slot = links[link_id]
                inputs[name] = [str(source_id), source_slot]
            elif ui_input is not None and ui_input.get("widget") is None and "link" in ui_input:
                # An explicit null link means the user disconnected this input.
                # Widget inputs also carry ``link: null`` when nothing is wired
                # into them; their value lives in ``widgets_values`` and must
                # still be emitted.
                continue
            elif name in widgets:
                inputs[name] = widgets[name]
            elif isinstance(definition, list) and definition:
                default = definition[0]
                if not isinstance(default, dict) or "default" in default:
                    if isinstance(default, dict):
                        default = default.get("default")
                    if default is not None and not _definition_is_forced_input(definition):
                        inputs[name] = default
            elif isinstance(definition, dict) and "default" in definition:
                inputs[name] = definition["default"]

        for name, ui_input in ui_inputs.items():
            if name in inputs or name == "upload":
                continue
            link_id = ui_input.get("link")
            if link_id is not None:
                if link_id not in links:
                    raise WorkflowError(
                        f"dangling dynamic link {link_id!r} in {node_type} node {node_id}"
                    )
                source_id, source_slot = links[link_id]
                inputs[name] = [str(source_id), source_slot]

        api_node: dict[str, Any] = {"class_type": node_type, "inputs": inputs}
        title = node.get("title")
        if title:
            api_node["_meta"] = {"title": title}
        prompt[node_id] = api_node

    validate_api(prompt)
    return prompt


def _definition_is_forced_input(definition: list[Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("forceInput") is True
        for item in definition
    )


def validate_api(workflow: dict[str, Any]) -> None:
    if not _looks_like_api(workflow):
        raise WorkflowError("API workflow must be a non-empty node-id object")
    for node_id, node in workflow.items():
        if not str(node_id):
            raise WorkflowError("API workflow has an empty node ID")
        cls = node.get("class_type")
        if not isinstance(cls, str) or not cls:
            raise WorkflowError(f"API node {node_id} has no class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise WorkflowError(f"API node {node_id} has invalid inputs")
        for name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2:
                source_id = str(value[0])
                if source_id not in workflow:
                    raise WorkflowError(
                        f"API node {node_id} input {name} references missing node "
                        f"{source_id}"
                    )


def validate_workflow(workflow: dict[str, Any]) -> None:
    if detect_format(workflow) == "api":
        validate_api(workflow)
        return
    node_ids = set(_nodes(workflow))
    for link in workflow.get("links", []):
        if not isinstance(link, list) or len(link) < 5:
            raise WorkflowError(f"invalid UI link: {link!r}")
        if str(link[1]) not in node_ids or str(link[3]) not in node_ids:
            raise WorkflowError(f"UI link references a missing node: {link!r}")


def iter_api_nodes(
    workflow: dict[str, Any], node_type: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    for node_id, node in workflow.items():
        if _node_type(node) == node_type:
            yield str(node_id), node
