"""Opt-in, rank-zero, write-once captures of real H3 numerics. No math changes."""
import json
import os
from pathlib import Path
import time

_SAVED = set()
_ATTACHED = set()
_DIT_CALLS = 0
_MAX_FILE_BYTES = 512 * 1024 * 1024


def _enabled():
    return bool(os.environ.get("VLLM_OMNI_910A_AUDIT_DIR"))


def _rank_zero():
    import torch
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0


def _tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def save_once(name, payload):
    """Metadata appears only after a complete, bounded write of detached tensors."""
    if not _enabled() or not _rank_zero() or name in _SAVED:
        return False
    import torch
    raw = Path(os.environ["VLLM_OMNI_910A_AUDIT_DIR"])
    root = raw.resolve(strict=True)
    if not root.is_dir() or not root.name.startswith("h3_round54_capture"):
        raise RuntimeError("Unexpected audit destination")
    if name not in {"encoder_input", "embedding", "vision_outputs", "text_inputs", "text_rotary",
                    "text_final", "text_attention0", "dit_attention0", "dit_attention49"} and not name.startswith("text_layer_"):
        raise RuntimeError("Unexpected capture name")
    size = _tensor_bytes(payload)
    if size > _MAX_FILE_BYTES:
        raise RuntimeError("Capture exceeds its explicit byte limit")
    target = root / (name + ".pt")
    metadata = root / (name + ".json")
    if target.exists() or metadata.exists():
        raise RuntimeError("Refuse to overwrite earlier audit evidence")
    start = time.perf_counter()
    copied = _cpu_copy(payload)
    with target.open("xb") as f:
        torch.save(copied, f)
    with metadata.open("x") as f:
        json.dump({"name": name, "tensor_bytes": size, "file_bytes": target.stat().st_size,
                   "capture_wall_s": time.perf_counter() - start,
                   "scope": "Rank-zero diagnostic copying synchronizes and is not benchmark timing"}, f, indent=2)
    _SAVED.add(name)
    print("[H3AUDIT] saved " + name, flush=True)
    return True


def _output_hook(name):
    def hook(module, args, output):
        save_once(name, output)
        return None
    return hook


def install_encoder_hooks(encoder, input_ids, vision_kwargs):
    if not _enabled() or not _rank_zero():
        return
    save_once("encoder_input", {"input_ids": input_ids, **vision_kwargs})
    tag = id(encoder)
    if tag in _ATTACHED:
        return
    _ATTACHED.add(tag)
    encoder.text_model.embed_tokens.register_forward_hook(_output_hook("embedding"))
    encoder.vision.register_forward_hook(_output_hook("vision_outputs"))

    def text_pre(mod, args, kwargs):
        save_once("text_inputs", {"inputs_embeds": args[0], "positions": args[1], **kwargs})
    encoder.text_model.register_forward_pre_hook(text_pre, with_kwargs=True)
    encoder.text_model.rotary_emb.register_forward_hook(_output_hook("text_rotary"))
    for index in (0, 1, 2, 9, 24, 49):
        encoder.text_model.layers[index].register_forward_hook(_output_hook("text_layer_%02d" % index))
    encoder.text_model.register_forward_hook(_output_hook("text_final"))


def capture_attention(q, k, v, out, *, kind, scale=None, n_pad=0):
    global _DIT_CALLS
    if not _enabled() or not _rank_zero():
        return
    ordinal = 0
    if kind == "dit":
        if q.ndim != 4 or q.shape[2] < 16384 or q.shape[-1] != 128:
            return
        ordinal = _DIT_CALLS
        _DIT_CALLS += 1
        if ordinal not in (0, 49):
            return
        name = "dit_attention%d" % ordinal
    else:
        name = "text_attention0"
    if name in _SAVED:
        return
    save_once(name, {"query": q, "key": k, "value": v, "output": out,
                     "scale": q.shape[-1] ** -0.5 if scale is None else float(scale),
                     "n_pad": int(n_pad), "causal": kind == "text", "matching_call_ordinal": ordinal})
