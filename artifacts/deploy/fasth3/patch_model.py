#!/usr/bin/env python3
"""Patch vllm_omni's native MiniMax-H3 transformer for FastH3 PDD inference.

Adds:
  * ``pdd_steps`` to the architecture config;
  * ``MiniMaxH3PDDHead``, a replicated final projection widened to
    ``pdd_steps`` heads that can emit the fused (weighted-mean) prediction of
    one fine-grid block per call;
  * final-layer / model plumbing (``pdd_step`` kwarg + ``prepare_pdd``).

The native parameter names (``final_layer.video_out.weight`` etc.) are kept so
the existing diffusers/native weight loader and the fp32 parameter guard keep
working unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else
              "/data/h3fast/ref/vllm-omni/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py")

ARCH_OLD = """    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
"""
ARCH_NEW = """    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # FastH3/PDD: number of heads in the widened final projection (0 = the
    # original single-head MiniMax-H3 projection).  A positive value makes the
    # final layer emit the fused prediction of one fine-grid block per call.
    pdd_steps: int = 0
"""

HEAD_CLASS = '''class MiniMaxH3PDDHead(nn.Module):
    """Replicated final projection widened to ``pdd_steps`` PDD heads.

    Head ``j`` predicts the rectified-flow direction ``q_j = eps - x0`` frozen on
    fine-grid interval ``j``.  The runtime sampler evaluates the transformer once
    per *block* of ``pdd_steps / num_steps`` consecutive heads and needs only the
    integration-weighted mean of those heads, so :meth:`prepare` precomputes the
    fused ``[num_steps * head_rows]`` weights and :meth:`forward` applies one.

    The projection is replicated (not tensor-parallel): the fused output is only
    ``head_rows`` wide, so all-gathering it would cost more than recomputing it,
    and the native downstream code already expects the full (gathered) width.
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        head_rows: int,
        *,
        modality: str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = int(arch.pdd_steps)
        self.head_rows = int(head_rows)
        self.modality = modality
        self.prefix = prefix
        self.weight = nn.Parameter(
            torch.empty(self.num_heads * self.head_rows, arch.hidden_size, dtype=_FP32_DTYPE),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(self.num_heads * self.head_rows, dtype=_FP32_DTYPE),
            requires_grad=False,
        )
        self._fused_weight: torch.Tensor | None = None
        self._fused_bias: torch.Tensor | None = None
        self._num_steps: int | None = None

    def prepare(self, plan) -> None:
        """Precompute the fused weight/bias of every runtime step."""
        from .pdd import fused_block_heads

        fused_weight = fused_block_heads(
            self.weight.detach(),
            num_heads=self.num_heads,
            head_rows=self.head_rows,
            plan=plan,
            modality=self.modality,
        )
        fused_bias = fused_block_heads(
            self.bias.detach().unsqueeze(-1),
            num_heads=self.num_heads,
            head_rows=self.head_rows,
            plan=plan,
            modality=self.modality,
        ).squeeze(-1)
        self._fused_weight = fused_weight.to(device=self.weight.device, dtype=_FP32_DTYPE)
        self._fused_bias = fused_bias.to(device=self.bias.device, dtype=_FP32_DTYPE)
        self._num_steps = int(plan.num_steps)

    def forward(self, x: torch.Tensor, step: int | None = None) -> torch.Tensor:
        if self._fused_weight is None:
            raise RuntimeError(
                f"PDD head {self.prefix} has no fused weights yet; "
                "call MiniMaxH3DiTModel.prepare_pdd(plan) first"
            )
        if step is None:
            if self._num_steps != 1:
                raise ValueError(f"PDD head {self.prefix} requires a step index")
            step = 0
        if not 0 <= int(step) < int(self._num_steps):
            raise ValueError(f"pdd step {step} out of range [0, {self._num_steps})")
        start = int(step) * self.head_rows
        weight = self._fused_weight[start:start + self.head_rows]
        bias = self._fused_bias[start:start + self.head_rows]
        return F.linear(x, weight, bias)


'''

INIT_OLD = """        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            video_patch_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.video_out",
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latents_dim,
            bias=True,
            gather_output=True,
            params_dtype=_FP32_DTYPE,
            quant_config=None,
            prefix=f"{prefix}.audio_out",
        )
"""
INIT_NEW = """        if arch.pdd_steps:
            self.video_out = MiniMaxH3PDDHead(
                arch, video_patch_dim, modality="video", prefix=f"{prefix}.video_out"
            )
            self.audio_out = MiniMaxH3PDDHead(
                arch, arch.audio_latents_dim, modality="audio", prefix=f"{prefix}.audio_out"
            )
        else:
            self.video_out = ColumnParallelLinear(
                arch.hidden_size,
                video_patch_dim,
                bias=True,
                gather_output=True,
                params_dtype=_FP32_DTYPE,
                quant_config=None,
                prefix=f"{prefix}.video_out",
            )
            self.audio_out = ColumnParallelLinear(
                arch.hidden_size,
                arch.audio_latents_dim,
                bias=True,
                gather_output=True,
                params_dtype=_FP32_DTYPE,
                quant_config=None,
                prefix=f"{prefix}.audio_out",
            )
"""

FWD_OLD = """    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:"""
FWD_NEW = """    def forward(
        self,
        x: torch.Tensor,
        *,
        t_emb: torch.Tensor,
        inverse_indices: torch.Tensor,
        pdd_step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:"""

APPLY_OLD = """        h = h.to(_FP32_DTYPE)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio
"""
APPLY_NEW = """        h = h.to(_FP32_DTYPE)
        if isinstance(self.video_out, MiniMaxH3PDDHead):
            return self.video_out(h, pdd_step), self.audio_out(h, pdd_step)
        video, _ = self.video_out(h)
        audio, _ = self.audio_out(h)
        return video, audio
"""

KWARGS_OLD = '''        "video_token_layout",
        "rope_table",
    }
)'''
KWARGS_NEW = '''        "video_token_layout",
        "rope_table",
        "pdd_step",
    }
)'''

CALL1_OLD = """            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=inverse_indices,
            )"""
CALL1_NEW = """            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=inverse_indices,
                pdd_step=pdd_step,
            )"""

CALL2_OLD = """            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=local_inverse_indices,
            )"""
CALL2_NEW = """            video_logits, audio_logits = self.final_layer(
                hidden,
                t_emb=t_emb,
                inverse_indices=local_inverse_indices,
                pdd_step=pdd_step,
            )"""

PREPARE_ANCHOR = """    def post_load_weights(self) -> None:"""
PREPARE_NEW = '''    def prepare_pdd(self, plan) -> None:
        """Prepare the FastH3 PDD sampler on this model.

        *plan* is a :class:`PDDSamplingPlan`.  Precomputes the fused per-step
        final-projection weights; the denoise loop then calls
        ``forward(pdd_step=step)`` once per runtime step.
        """
        if not self.arch.pdd_steps:
            raise ValueError("prepare_pdd requires a checkpoint with pdd_steps > 0")
        self._pdd_plan = plan
        for module in self.modules():
            if isinstance(module, MiniMaxH3PDDHead):
                module.prepare(plan)

    @property
    def pdd_plan(self):
        return getattr(self, "_pdd_plan", None)

    def post_load_weights(self) -> None:'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"anchor {label!r} matched {count} times (expected 1)")
    return text.replace(old, new, 1)


def main() -> None:
    source = TARGET.read_text()
    if "MiniMaxH3PDDHead" in source:
        raise SystemExit("already patched")
    out = replace_once(source, ARCH_OLD, ARCH_NEW, "arch.pdd_steps")
    out = replace_once(out, "class MiniMaxH3FinalLayer(nn.Module):", HEAD_CLASS + "class MiniMaxH3FinalLayer(nn.Module):", "insert PDD head")
    out = replace_once(out, INIT_OLD, INIT_NEW, "final layer init")
    out = replace_once(out, FWD_OLD, FWD_NEW, "final layer forward signature")
    out = replace_once(out, APPLY_OLD, APPLY_NEW, "final layer apply")
    out = replace_once(out, KWARGS_OLD, KWARGS_NEW, "forward kwargs")
    out = replace_once(out, CALL1_OLD, CALL1_NEW, "final layer call (gathered)")
    out = replace_once(out, CALL2_OLD, CALL2_NEW, "final layer call (sharded)")
    out = replace_once(out, PREPARE_ANCHOR, PREPARE_NEW, "prepare_pdd")
    # Read the pdd step once at the top of the model forward.
    anchor = """        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")"""
    new_anchor = """        pdd_step = kwargs.get("pdd_step")
        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")"""
    out = replace_once(out, anchor, new_anchor, "read pdd_step")
    TARGET.write_text(out)
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
