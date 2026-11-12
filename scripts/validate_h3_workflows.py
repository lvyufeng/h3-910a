#!/usr/bin/env python3
"""Validate the four checked-in MiniMax H3 ComfyUI workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_MODELS = {
    "UNETLoader": (
        "unet_name",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ),
    "CLIPLoader": (
        "clip_name",
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ),
    "MiniMaxH3TurboLoRA": (
        "lora_name",
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    ),
}
EXPECTED_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
EXPECTED_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


@dataclass(frozen=True)
class Case:
    instance: int
    phase: str
    width: int
    height: int
    frames: int


CASES = (
    Case(1, "smoke", 608, 352, 90),
    Case(1, "target", 864, 480, 124),
    Case(2, "smoke", 608, 352, 90),
    Case(2, "target", 864, 480, 124),
)


class ValidationError(ValueError):
    """Raised when an H3 workflow violates its execution contract."""


def load_ui_workflow(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise ValidationError(f"{path}: expected a ComfyUI UI workflow")
    return data


def node_map(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node in workflow["nodes"]:
        if not isinstance(node, dict) or "id" not in node:
            raise ValidationError("workflow contains a malformed node")
        node_id = str(node["id"])
        if node_id in result:
            raise ValidationError(f"duplicate node id {node_id}")
        result[node_id] = node
    return result


def one_node(
    nodes: dict[str, dict[str, Any]], node_type: str
) -> dict[str, Any]:
    matches = [node for node in nodes.values() if node.get("type") == node_type]
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one {node_type} node, found {len(matches)}"
        )
    return matches[0]


def ui_input(node: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        value
        for value in node.get("inputs", [])
        if isinstance(value, dict) and value.get("name") == name
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"{node.get('type')} node {node.get('id')}: "
            f"expected one {name!r} input, found {len(matches)}"
        )
    return matches[0]


def widget_value(node: dict[str, Any], name: str) -> Any:
    values = node.get("widgets_values")
    inputs = node.get("inputs", [])
    if isinstance(values, dict):
        if name not in values:
            raise ValidationError(
                f"{node.get('type')} node {node.get('id')}: "
                f"missing widget {name!r}"
            )
        return values[name]
    widget_names = [
        value.get("name")
        for value in inputs
        if isinstance(value, dict)
        and isinstance(value.get("widget"), dict)
        and value.get("name")
    ]
    if name not in widget_names or not isinstance(values, list):
        raise ValidationError(
            f"{node.get('type')} node {node.get('id')}: "
            f"missing widget {name!r}"
        )
    index = widget_names.index(name)
    if index >= len(values):
        raise ValidationError(
            f"{node.get('type')} node {node.get('id')}: "
            f"widget {name!r} has no value"
        )
    return values[index]


def require_equal(
    path: Path, description: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        raise ValidationError(
            f"{path}: {description} is {actual!r}, expected {expected!r}"
        )


def require_number(
    path: Path, description: str, actual: Any, expected: int | float
) -> None:
    try:
        number = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{path}: {description} is not numeric: {actual!r}"
        ) from exc
    if number != float(expected):
        raise ValidationError(
            f"{path}: {description} is {actual!r}, expected {expected!r}"
        )


def validate_links(workflow: dict[str, Any]) -> None:
    nodes = node_map(workflow)
    links = workflow.get("links")
    if not isinstance(links, list):
        raise ValidationError("workflow links must be a list")
    link_ids: set[Any] = set()
    for link in links:
        if not isinstance(link, list) or len(link) < 5:
            raise ValidationError(f"malformed link: {link!r}")
        link_id, source_id, source_slot, target_id, target_slot = link[:5]
        if link_id in link_ids:
            raise ValidationError(f"duplicate link id {link_id}")
        link_ids.add(link_id)
        if str(source_id) not in nodes or str(target_id) not in nodes:
            raise ValidationError(f"link {link_id} references a missing node")
        source = nodes[str(source_id)]
        target = nodes[str(target_id)]
        outputs = source.get("outputs", [])
        inputs = target.get("inputs", [])
        if (
            not isinstance(source_slot, int)
            or source_slot < 0
            or source_slot >= len(outputs)
        ):
            raise ValidationError(f"link {link_id} has invalid source slot")
        if (
            not isinstance(target_slot, int)
            or target_slot < 0
            or target_slot >= len(inputs)
        ):
            raise ValidationError(f"link {link_id} has invalid target slot")
        if inputs[target_slot].get("link") != link_id:
            raise ValidationError(
                f"link {link_id} is not attached to target input slot {target_slot}"
            )
        if link_id not in (outputs[source_slot].get("links") or []):
            raise ValidationError(
                f"link {link_id} is not attached to source output slot {source_slot}"
            )


def validate_case(path: Path, case: Case) -> None:
    workflow = load_ui_workflow(path)
    validate_links(workflow)
    nodes = node_map(workflow)

    video = one_node(nodes, "MiniMaxH3ImageToVideo")
    for name, expected in (
        ("width", case.width),
        ("height", case.height),
        ("length", case.frames),
    ):
        require_number(path, f"video {name}", widget_value(video, name), expected)
        require_equal(
            path,
            f"video {name} input link",
            ui_input(video, name).get("link"),
            None,
        )
    for name in ("first_frame", "last_frame"):
        require_equal(
            path,
            f"video {name} input link",
            ui_input(video, name).get("link"),
            None,
        )

    parallel = one_node(nodes, "MultiNPUParallelConfig")
    require_number(
        path, "parallel npu_count", widget_value(parallel, "npu_count"), 4
    )

    scheduler = one_node(nodes, "BasicScheduler")
    require_number(path, "scheduler steps", widget_value(scheduler, "steps"), 4)

    create_video = one_node(nodes, "CreateVideo")
    require_number(path, "CreateVideo fps", widget_value(create_video, "fps"), 24)

    expected_prefix = f"h3-910a/instance-{case.instance}/{case.phase}"
    save_video = one_node(nodes, "SaveVideo")
    require_equal(
        path,
        "SaveVideo filename_prefix",
        widget_value(save_video, "filename_prefix"),
        expected_prefix,
    )

    for node_type, (widget_name, expected) in EXPECTED_MODELS.items():
        node = one_node(nodes, node_type)
        require_equal(
            path,
            f"{node_type} {widget_name}",
            widget_value(node, widget_name),
            expected,
        )

    vae_loaders = sorted(
        (
            str(node.get("id")),
            widget_value(node, "vae_name"),
        )
        for node in nodes.values()
        if node.get("type") == "VAELoader"
    )
    if len(vae_loaders) != 2:
        raise ValidationError(
            f"{path}: expected two VAELoader nodes, found {len(vae_loaders)}"
        )
    expected_vaes = sorted((loader[0], loader[1]) for loader in vae_loaders)
    actual_vae_names = sorted(loader[1] for loader in vae_loaders)
    require_equal(
        path,
        "VAELoader model set",
        actual_vae_names,
        sorted((EXPECTED_VIDEO_VAE, EXPECTED_AUDIO_VAE)),
    )

    parallel_inputs = parallel.get("inputs", [])
    role_sources: dict[str, str] = {}
    for link in workflow["links"]:
        if str(link[3]) != str(parallel["id"]):
            continue
        slot = link[4]
        if isinstance(slot, int) and 0 <= slot < len(parallel_inputs):
            role = parallel_inputs[slot].get("name")
            if role in {"video_vae", "audio_vae"}:
                role_sources[role] = str(link[1])
    require_equal(
        path,
        "video VAE source",
        next(
            (loader[1] for loader in vae_loaders if loader[0] == role_sources.get("video_vae")),
            None,
        ),
        EXPECTED_VIDEO_VAE,
    )
    require_equal(
        path,
        "audio VAE source",
        next(
            (loader[1] for loader in vae_loaders if loader[0] == role_sources.get("audio_vae")),
            None,
        ),
        EXPECTED_AUDIO_VAE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the four MiniMax H3 instance workflows."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/design/workflows"),
        help="directory containing instance-{1,2}-{smoke,target}.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    for case in CASES:
        path = args.root / f"instance-{case.instance}-{case.phase}.json"
        try:
            validate_case(path, case)
        except ValidationError as exc:
            failures.append(str(exc))
        else:
            print(f"PASS {path}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        return 1
    print(f"PASS all {len(CASES)} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
