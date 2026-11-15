#!/usr/bin/env python3
"""Summarize the successful real-weight VAE encode without a steady-time claim."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = json.loads((root / "vae_encode02/result.json").read_text())
assert "failure" not in raw
assert len(raw["cases"]) == 2
assert all(x["finite"] and x["output_dtype"] == "torch.float32" for x in raw["cases"])
assert raw["cases"][0]["repeat_bit_exact"]
large = raw["cases"][1]
assert [large["width"], large["height"], large["output_shape"]] == [1344, 768, [1008, 96]]
trace = raw["padding_trace"]
packed = [x for x in trace if x["input_format"] == 32]
assert packed and all(x["output_format"] == 30 for x in packed)
summary = {"encoder_dtype": raw["encoder_dtype"], "parameter_bytes": raw["parameter_bytes"],
           "observed_calls": raw["cases"], "padding_calls_recorded": len(trace),
           "packed_to_base_padding_calls": len(packed), "all_packed_padding_outputs_format": 30,
           "loaded_conv_source_sha256": raw["loaded_conv_source_sha256"],
           "qualification": "single observed real-weight calls include cold-path effects; not steady-state or VAE decode benchmarks"}
out = root / "vae_encode_summary.json"
if out.exists():
    raise RuntimeError("Refuse to overwrite an existing summary")
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
