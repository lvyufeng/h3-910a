#!/usr/bin/env python3
"""Run the real VAE image encoder alone, preserving FP32 weights and encode rules.

Avoids paying for a full Qwen/DiT engine while validating a local pad repair.
Uses actual checkpoint encoder weights, reference encode_image and native seed/
normalization, never a dummy encoder or a computation bypass.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from PIL import Image, ImageDraw
import torch
import torch_npu

ROOT = Path(os.environ.get("H3_VAE_PROBE_DIR", "/tmp/h3_round51_vae_encode01"))
if ROOT.exists():
    raise RuntimeError("Refuse to overwrite prior VAE encode evidence")
ROOT.mkdir()
torch.npu.set_device(0)
from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

result = {"component_path": "/home/mseco/models/MiniMax-H3-fp16/FL2VA/video_vae",
          "device": "npu:0", "encoder_dtype": "float32", "encode_only": True, "cases": []}
vae = None
try:
    start = time.perf_counter()
    vae = MiniMaxH3VideoVAE(result["component_path"], device=torch.device("npu:0"),
                           load_device=torch.device("cpu"), encode_only=True, trust_remote_code=True)
    result["cpu_load_seconds"] = time.perf_counter() - start
    result["parameter_bytes"] = sum(p.numel() * p.element_size() for p in vae.parameters())
    assert all(p.dtype == torch.float32 for p in vae.parameters())
    assert not hasattr(vae.model, "decoder"), "Encoder-only loader included a decoder"
    down = next(m for m in vae.model.encoder.modules() if type(m).__name__ == "Downsample3D")
    module = sys.modules[type(down).__module__]
    assert "_pad_base_format" in inspect.getsource(type(down).forward), "Loaded VAE did not include the checked patch"
    result["loaded_vae_cnn"] = str(Path(inspect.getfile(type(down))).resolve())
    conv_module = sys.modules[type(down.conv).__module__]
    assert "_pad_base_format" in inspect.getsource(conv_module.BaseConv3d._apply_padding), "Loaded Conv3D missed the checked layout fix"
    result["loaded_conv"] = str(Path(inspect.getfile(type(down.conv))).resolve())
    result["loaded_conv_source_sha256"] = hashlib.sha256(Path(result["loaded_conv"]).read_bytes()).hexdigest()
    native_pad = module._pad_base_format
    native_conv_pad = conv_module._pad_base_format
    trace = []
    def record_pad(x, pad, mode, native=native_pad, caller="vae_cnn"):
        before = int(torch_npu.get_npu_format(x)) if x.device.type == "npu" else None
        y = native(x, pad, mode)
        trace.append({"caller": caller, "input_shape": list(x.shape), "input_dtype": str(x.dtype), "mode": mode,
                      "input_format": before,
                      "output_format": int(torch_npu.get_npu_format(y)) if y.device.type == "npu" else None})
        return y
    module._pad_base_format = record_pad
    conv_module._pad_base_format = lambda x, pad, mode: record_pad(x, pad, mode, native=native_conv_pad, caller="conv")
    start = time.perf_counter()
    vae.encoder_component.load_to_device()
    torch.npu.synchronize()
    result["stage_to_device_seconds"] = time.perf_counter() - start
    print("Real FP32 encoder staged; params_MB=%.1f" % (result["parameter_bytes"] / 1024**2), flush=True)
    sizes = [(64, 64)]
    if os.environ.get("H3_VAE_TEST_FULL_IMAGE", "0") == "1":
        sizes.append((1344, 768))
    for width, height in sizes:
        image = Image.new("RGB", (width, height), (200, 205, 212))
        d = ImageDraw.Draw(image)
        d.rectangle((0, height * 5 // 6, width, height), fill=(98, 102, 108))
        r = max(4, width // 30)
        d.ellipse((width // 3 - r, height * 2 // 3 - r, width // 3 + r, height * 2 // 3 + r), fill=(215, 30, 30))
        input_path = ROOT / f"image_{width}x{height}.png"
        image.save(input_path)
        start = time.perf_counter()
        rows = vae.encode_image(image)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        expected = (height // 32 * (width // 32), 4 * int(vae.config_dict["latent_channels"]))
        case = {"width": width, "height": height, "encode_seconds": elapsed,
                "output_shape": list(rows.shape), "expected_shape": list(expected),
                "output_dtype": str(rows.dtype), "output_device": str(rows.device),
                "finite": bool(torch.isfinite(rows).all()),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}
        assert tuple(rows.shape) == expected, case
        assert rows.dtype == torch.float32 and rows.device.type == "cpu" and case["finite"], case
        torch.save(rows, ROOT / f"rows_{width}x{height}.pt")
        if width == 64:
            repeat = vae.encode_image(image)
            torch.npu.synchronize()
            case["repeat_bit_exact"] = bool(torch.equal(rows, repeat))
            case["repeat_max_abs_diff"] = (rows - repeat).abs().max().item()
            assert case["repeat_bit_exact"], case
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
    result["padding_trace"] = trace
    assert any(t["input_format"] == 32 and t["output_format"] == 30 for t in trace), "Actual encode did not exercise packed reflection padding"
except BaseException as exc:
    result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    raise
finally:
    if "trace" in locals():
        result["padding_trace"] = trace
    if vae is not None:
        try:
            vae.encoder_component.offload_to_cpu()
        except Exception as exc:
            result["cleanup_warning"] = str(exc)
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "padding_trace"}
    summary["padding_call_count"] = len(result.get("padding_trace", []))
    print(json.dumps(summary, indent=2), flush=True)
