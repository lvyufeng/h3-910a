#!/usr/bin/env python3
"""Build the Round79 diagnostic FP16 encoder candidate from the actual R76 source.

The candidate is generated, never hand-edited: this script reads the verified
production encoder read-only, applies a fixed list of line-range edits, and
writes the result plus a manifest recording the original/candidate SHA256 and
every touched line span. Regenerating from the same origin therefore reproduces
the candidate byte for byte, and any edit outside the declared spans is
impossible by construction.

The candidate is diagnostics only:

* it is never imported by production code; the root driver loads it by path;
* it has not been executed on 910A and cannot establish an NPU precision pass;
* it does not consume the Round78 CPU FP16 reference, which is still running as
  its own job (a completed CPU gold is required, and casting old BF16 outputs to
  FP16 is explicitly not a reference).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CANDIDATE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CANDIDATE_DIR.parents[3]
SOURCE = PROJECT_ROOT / "artifacts/diagnostics/round76/source/encoder.py"
CANDIDATE = CANDIDATE_DIR / "candidate_encoder79.py"
MANIFEST = CANDIDATE_DIR / "build_manifest79.json"

# Pinned by Round76 final verification and re-checked here before any edit.
SOURCE_SHA256 = "7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d"

# Edit order is irrelevant to the result: edits are applied bottom-up so the
# remaining original line numbers stay valid. ``start``/``end`` are 1-based
# inclusive spans of the ORIGINAL file; ``anchor`` must appear in the first
# original line of the span and guards against an unexpected source revision.
EDITS: list[dict[str, object]] = [
    {
        "id": "E0-docstring-dtype",
        "start": 14,
        "end": 16,
        "anchor": "plain BF16 weights",
        "purpose": "The module docstring must not claim plain BF16 weights once the candidate also supports FP16.",
        "replacement": """- dense parameters in the resolved compute dtype (BF16 by default, FP16 in the
  Round79 diagnostic candidate), only the first
  ``MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER`` decoder layers are kept (the
  checkpoint consumes the unnormalized hidden state after decoder layer 49);
""",
    },
    {
        "id": "E1-embedding-reduce-fp16",
        "start": 381,
        "end": 388,
        "anchor": "def _text_hf_embedding_all_reduce",
        "purpose": (
            "Add the dtype resolver and promote the vocabulary-embedding all-reduce to FP32 for FP16, "
            "matching the single-device reference sum; the BF16 branch keeps its existing gate."
        ),
        "replacement": '''def _r79_compute_dtype() -> torch.dtype:
    """Resolve the encoder compute dtype for the Round79 diagnostic candidate.

    Round78 established that the official Qwen3-VL checkpoint is BF16-stored and
    that no suitable full-stack FP16 gold was found in the bounded search; this
    candidate is compared against an independently recomputed CPU FP16 reference. The mode is explicit:
    only the historical ``fp16``/``half`` setting selects FP16, and the stored
    parameters and every encoder I/O boundary then share that single dtype.
    """
    import os
    if os.environ.get("VLLM_OMNI_910A_TE_DTYPE", "").lower() in ("fp16", "half"):
        return torch.float16
    return torch.bfloat16


def _text_hf_embedding_all_reduce(group: Any, value: torch.Tensor) -> torch.Tensor:
    if value.device.type in ("npu", "privateuseone") and value.dtype == torch.float16:
        # The single-device reference sums the whole vocabulary row inside one
        # embedding lookup, so an FP16 partial all-reduce would introduce a
        # rounding step with no counterpart in the CPU gold. Promote the
        # collective to FP32 and round once, like the row-parallel projections.
        return group.all_reduce(value.float()).to(value.dtype)
    if (_text_hf_math_enabled() and value.device.type in ("npu", "privateuseone")
            and value.dtype == torch.bfloat16):
        # Only the vocabulary embedding reaches the common coordinator as
        # BF16. Text row projections already promote their partials to FP32.
        # Preserve the embedding's BF16 range instead of a FP16 round trip.
        return group.all_reduce(value.float()).to(value.dtype)
    return group.all_reduce(value)
''',
    },
    {
        "id": "E2-rmsnorm-fp16-gamma",
        "start": 408,
        "end": 428,
        "anchor": "    def forward_npu(",
        "purpose": (
            "Add the FP16 RMSNorm branch with the reference's two rounding points (FP32 reduction, "
            "cast to FP16, FP16 gamma multiply). The BF16 block is preserved verbatim."
        ),
        "replacement": '''    def forward_npu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if (x.device.type in ("npu", "privateuseone") and x.dtype == torch.float16
                and self.weight.dtype == torch.float16):
            # Round79 FP16 path: reproduce ``Qwen3VLTextRMSNorm.forward`` as the
            # reference writes it, which rounds twice. The reference normalises
            # in FP32, casts the normalised value to the activation dtype, and
            # only then multiplies by the same-dtype gamma. Folding gamma into
            # the FP32 point math would round once instead of twice and is
            # therefore not the reference arithmetic.
            input_dtype = x.dtype
            with torch.autocast(device_type=x.device.type, enabled=False):
                if residual is not None:
                    # The reference adds the residual in the activation dtype
                    # before the norm. A single correctly rounded add is that
                    # same operation, so this introduces no FP32 island.
                    residual = residual + x
                    x = residual
                xf = x.float()
                variance = xf.square().mean(-1, keepdim=True)
                normalized = (xf * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)
                output = self.weight * normalized
            return (output, residual) if residual is not None else output
        if not (_text_hf_math_enabled() and x.device.type in ("npu", "privateuseone")
                and x.dtype == torch.bfloat16 and self.weight.dtype == torch.bfloat16):
            return super().forward_npu(x, residual)
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            # HF rounds the residual sum to BF16 before normalization, then
            # rounds normalized values before gamma multiplication. Use FP32
            # point math between casts, not a native BF16 multiply fallback.
            if residual is not None:
                residual = (residual.float() + x.float()).to(input_dtype)
                x = residual
            xf = x.float()
            variance = xf.square().mean(-1, keepdim=True)
            normalized = (xf * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)
            output = (normalized.float() * self.weight.float()).to(input_dtype)
        return (output, residual) if residual is not None else output
''',
    },
    {
        "id": "E3-text-rope-fp16",
        "start": 495,
        "end": 496,
        "anchor": '        if (_text_hf_math_enabled() and x.device.type in ("npu", "privateuseone")',
        "purpose": (
            "Route FP16 text M-RoPE through the FP32-frequency branch so cos/sin are cast to FP16 "
            "exactly as the reference returns them; the BF16 gate is unchanged."
        ),
        "replacement": '''        # Round79: the FP16 path uses this branch unconditionally. The reference
        # builds the frequencies as an FP32 outer product, applies the same
        # interleaved M-RoPE layout, and returns cos/sin cast to the activation
        # dtype. The direct multiply below is that outer product without the NPU
        # FP32-matmul frequency error, and the trailing casts already follow
        # ``x.dtype`` (FP16 in the diagnostic mode).
        if (x.device.type in ("npu", "privateuseone")
                and (x.dtype == torch.float16
                     or (_text_hf_math_enabled() and x.dtype == torch.bfloat16))):
''',
    },
    {
        "id": "E4-pos-interpolate-fp32",
        "start": 834,
        "end": 836,
        "anchor": '        npu_fp32_interpolation = (device.type in ("npu", "privateuseone")',
        "purpose": (
            "Use the reference's FP32 bilinear weights and FP32 four-term sum for both diagnostic dtypes "
            "instead of gating it behind VLLM_OMNI_910A_VISION_FP32."
        ),
        "replacement": '''        # Round79: the reference always interpolates with FP32 bilinear weights
        # and FP32 four-term sums, converting once back to the activation dtype,
        # so the FP16 diagnostics need this unconditionally. BF16 is now taken
        # unconditionally here too, which makes the BF16 candidate numerically
        # the more faithful form and NOT bit-identical to the production R76
        # default that gated this behind VLLM_OMNI_910A_VISION_FP32.
        npu_fp32_interpolation = (
            device.type in ("npu", "privateuseone")
            and self.pos_embed.weight.dtype in (torch.bfloat16, torch.float16)
        )
''',
    },
    {
        "id": "E5-encoder-dtype-resolution",
        "start": 1264,
        "end": 1272,
        "anchor": "        # Keep BF16 encoder storage by default.",
        "purpose": (
            "Replace the inline env parse with the single dtype resolver and record the resolved "
            "compute dtype plus the FP16-mode flag on the encoder."
        ),
        "replacement": '''        # Round79 diagnostic candidate: ``_r79_compute_dtype()`` is the single
        # source of truth for the stored parameters and for every encoder I/O
        # boundary. The production default stays BF16; the FP16 branch is only
        # taken when the mode is requested explicitly, and it is never combined
        # with BF16 intermediates.
        dtype = _r79_compute_dtype()
        self.compute_dtype = dtype
        self._r79_fp16 = dtype == torch.float16
''',
    },
    {
        "id": "E6-encoder-dtype-defaults",
        "start": 1254,
        "end": 1254,
        "anchor": "        self._tp_size = 1",
        "purpose": "Define compute_dtype before the early return so unloaded instances still expose it.",
        "replacement": '''        self._tp_size = 1
        self.compute_dtype = torch.bfloat16
        self._r79_fp16 = False
''',
    },
    {
        "id": "E7-pixel-values-dtype",
        "start": 1453,
        "end": 1453,
        "anchor": "                pixel_values.to(device, torch.bfloat16),",
        "purpose": "Take the image input dtype from the stored vision parameters instead of a hardcoded BF16 cast.",
        "replacement": "                pixel_values.to(device, self._r79_io_dtype()),\n",
    },
    {
        "id": "E8-pixel-videos-dtype",
        "start": 1461,
        "end": 1461,
        "anchor": "                pixel_values_videos.to(device, torch.bfloat16),",
        "purpose": "Same boundary fix for the video pixels.",
        "replacement": "                pixel_values_videos.to(device, self._r79_io_dtype()),\n",
    },
    {
        "id": "E9-hidden-return-dtype",
        "start": 1492,
        "end": 1492,
        "anchor": "        return hidden.to(torch.bfloat16)",
        "purpose": (
            "Return the graph's own dtype. The reference has no final BF16 hop, so the hardcoded "
            "cast was an encoder-specific extra rounding. Hidden is already the compute dtype, "
            "so the BF16 result is unchanged."
        ),
        "replacement": '''        # Round79: return the graph's own dtype. The single-device reference has
        # no final BF16 hop, so the previous hardcoded cast was an
        # encoder-specific rounding with no counterpart in the CPU gold. The
        # hidden state is already the compute dtype, so the BF16 result is
        # unchanged while FP16 no longer round-trips through BF16.
        return hidden
''',
    },
    {
        "id": "E10-io-dtype-and-self-check",
        "start": 1428,
        "end": 1428,
        "anchor": "    def _encode(",
        "purpose": (
            "Add the I/O dtype accessor and a fail-closed self-check so a mixed "
            "parameter dtype can never be silently benchmarked as one FP16 graph."
        ),
        "replacement": '''    def _r79_io_dtype(self) -> torch.dtype:
        """Dtype of the encoder's image/video input and hidden-state output.

        Derived from the stored vision parameters so the I/O boundary cannot
        drift away from the weights that consume it.
        """
        return self.vision.patch_embed.proj.weight.dtype

    def r79_self_check(self) -> dict[str, Any]:
        """Fail closed unless the candidate really is one coherent parameter dtype.

        The Round78 audit found that setting the historical half flag alone left
        BF16 pixel and hidden boundaries in place, i.e. an incoherent graph that
        no round had executed. This check makes that specific incoherence
        impossible to mistake for a clean FP16 run.
        """
        import collections

        io_dtype = self._r79_io_dtype()
        counts = collections.Counter(
            str(parameter.dtype)
            for module in (self.vision, self.text_model)
            for parameter in module.parameters()
        )
        details = {
            "compute_dtype": str(self.compute_dtype),
            "io_dtype": str(io_dtype),
            "vision_patch_embed_dtype": str(self.vision.patch_embed.proj.weight.dtype),
            "text_embedding_dtype": str(self.text_model.embed_tokens.weight.dtype),
            "parameter_dtype_counts": dict(counts),
            "fp16_mode": bool(self._r79_fp16),
        }
        if io_dtype != self.compute_dtype:
            raise RuntimeError(
                f"R79 encoder I/O dtype {io_dtype} does not match parameters {self.compute_dtype}"
            )
        expected = {str(self.compute_dtype)}
        if set(counts) != expected:
            raise RuntimeError(f"R79 encoder parameter dtypes are not uniform: {details}")
        return details

    def _encode(
''',
    },
    {
        "id": "E11-candidate-constants",
        "start": 52,
        "end": 52,
        "anchor": "MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120",
        "purpose": "Mark the module as the generated diagnostic candidate and pin the origin it was built from.",
        "replacement": '''MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120

# Round79 diagnostic candidate, generated from the verified production encoder by
# ``build_candidate79.py``. Production never imports this module: the root driver
# loads it by path after distributed init. It removes the three BF16 I/O
# conversions and makes the FP16 branches follow the reference HF rounding.
R79_CANDIDATE = True
R79_SOURCE_SHA256 = "7c22d98c28f4e93550fd3848bfc174121250259fdf50bcd2f7920f826acf311d"
''',
    },
]


def apply_edits(lines: list[str]) -> tuple[list[str], list[dict[str, object]]]:
    """Apply EDITS bottom-up, returning the new lines and the resolved spans."""
    out = list(lines)
    resolved = []
    for edit in sorted(EDITS, key=lambda item: int(item["start"]), reverse=True):
        start, end = int(edit["start"]), int(edit["end"])
        assert 1 <= start <= end <= len(out), f"{edit['id']}: span outside source"
        anchor = str(edit["anchor"])
        assert anchor in out[start - 1], (
            f"{edit['id']}: anchor {anchor!r} missing from original line {start}"
        )
        replacement = str(edit["replacement"]).splitlines(keepends=True)
        out[start - 1:end] = replacement
        resolved.append(
            {
                "id": edit["id"],
                "purpose": edit["purpose"],
                "original_lines": [start, end],
                "replacement_line_count": len(replacement),
            }
        )
    return out, resolved


def build() -> dict[str, object]:
    original_bytes = SOURCE.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    assert original_sha == SOURCE_SHA256, f"unexpected R76 encoder hash {original_sha}"

    original_lines = original_bytes.decode("utf-8").splitlines(keepends=True)
    new_lines, resolved = apply_edits(original_lines)
    candidate_text = "".join(new_lines)
    assert candidate_text != original_bytes.decode("utf-8")
    compile(candidate_text, str(CANDIDATE), "exec")
    CANDIDATE.write_text(candidate_text, encoding="utf-8")

    candidate_sha = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
    manifest = {
        "scope": "Generated Round79 diagnostic FP16 encoder candidate; never imported by production.",
        "generator": "build_candidate79.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_path": str(SOURCE.relative_to(PROJECT_ROOT)),
        "source_sha256": original_sha,
        "source_line_count": len(original_lines),
        "candidate_path": str(CANDIDATE.relative_to(PROJECT_ROOT)),
        "candidate_sha256": candidate_sha,
        "candidate_line_count": len(new_lines),
        "candidate_bytes": len(candidate_text.encode("utf-8")),
        "edit_count": len(resolved),
        "edits": sorted(resolved, key=lambda item: item["original_lines"][0]),
        "executed": False,
        "NPU_precision_established": False,
        "CPU_FP16_gold_consumed": False,
        "tolerance_set": False,
        "production_files_modified": [],
        "notes": [
            "Regenerating from the pinned source reproduces the candidate byte for byte.",
            "Every difference from the source lies inside a declared original_lines span.",
            "TEXT_PAD_KV / TEXT_FP32_LINEAR / _text_fp32_local_linear do not exist in this R76 source; "
            "they belong to the separate round65 candidate and needed no disabling here.",
            "BF16 branches are preserved verbatim; the interpolation edit intentionally moves BF16 to the "
            "reference-faithful FP32 form, so BF16-vs-production is not bit-identical.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = build()
    print(f"source  {result['source_sha256']}")
    print(f"candidate {result['candidate_sha256']}")
    print(f"edits {result['edit_count']}  lines {result['source_line_count']} -> {result['candidate_line_count']}")