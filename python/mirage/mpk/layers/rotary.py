"""Precomputed RoPE cos/sin tables.

No MPK task — the actual RoPE rotation runs inside the attention
kernel. This module owns the precomputed ``cos`` and ``sin`` tables as
``nn.Buffer`` (``register_buffer(..., persistent=False)``) so they
move with ``.to(device, dtype)``, stay out of ``state_dict()`` (HF
checkpoints don't ship RoPE tables), and are not seen as trainable.
``compile()`` returns ``(cos_dt, sin_dt)`` for the attention kernel to
consume as ``cos_pos_embed`` / ``sin_pos_embed``.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ._base import MPKModule
from ...core import DTensor


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (
        dim
        * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)


class RotaryEmbedding(MPKModule):
    """Precomputed cos/sin tables for RoPE.

    ``cos`` and ``sin``: ``(max_position_embeddings, head_dim)`` bf16
    non-persistent buffers, using the HF
    ``torch.cat((freqs, freqs), dim=-1)`` ``rotate_half`` convention.
    ``head_dim`` must be even.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        *,
        rope_scaling: Optional[dict] = None,
        interleaved: bool = False,
        prefix: str = "",
    ) -> None:
        """Precompute the RoPE cos/sin tables and register them as buffers.

        Tensor contract (set by ``__init__``; no task is emitted):
          cos: (max_position_embeddings, head_dim) bf16 non-persistent buffer.
          sin: (max_position_embeddings, head_dim) bf16 non-persistent buffer.

        Two layout conventions are supported:
          * ``interleaved=False`` (default): HF/LLaMA
            ``torch.cat((freqs, freqs), dim=-1)`` ``rotate_half`` convention —
            each (cos, sin) value goes to ``[..., :D/2]`` and ``[..., D/2:]``.
            Used by Qwen3-style attention kernels.
          * ``interleaved=True``: GPT-J / DeepSeek-MLA convention —
            ``repeat_interleave(2)`` layout ``[c0,c0,c1,c1,...]`` so the
            kernel rotating pairs ``(x[2i], x[2i+1])`` reads the matching
            angle. This is numerically equivalent (for QK dot products) to
            HF DeepSeek-V3 ``apply_rotary_pos_emb_interleave``.

        ``rope_scaling`` (when ``type/rope_type == "yarn"``) applies YARN
        long-context scaling to the inverse frequencies and an ``mscale``
        attention factor, matching HF ``_compute_yarn_parameters``.
        ``head_dim`` must be even.
        """
        super().__init__(prefix=prefix)
        if head_dim % 2 != 0:
            raise ValueError(
                f"RotaryEmbedding requires an even head_dim; got head_dim={head_dim}."
            )
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = float(base)
        self.rope_scaling = rope_scaling
        self.interleaved = interleaved

        cos, sin = self._precompute_freqs(
            head_dim=head_dim,
            max_pos=max_position_embeddings,
            base=self.base,
            rope_scaling=rope_scaling,
            interleaved=interleaved,
        )
        # persistent=False so the tables don't collide with HF state_dict
        # keys yet still migrate with .to(device, dtype).
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _precompute_freqs(
        head_dim: int,
        max_pos: int,
        base: float,
        rope_scaling: Optional[dict] = None,
        interleaved: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RoPE cos/sin precomputation in fp32, cast to bf16.

        Replicates the proven DeepSeek-V3 builder construction: plain or
        YARN-scaled inverse frequencies, an ``mscale`` attention factor, and
        either the ``cat`` (``interleaved=False``) or ``repeat_interleave(2)``
        (``interleaved=True``) angle layout.
        """
        half = head_dim // 2
        rope_params = rope_scaling or {}
        rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))

        pos_freqs = base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        if rope_type in ("yarn", "deepseek_yarn", "deepseek_llama_scaling"):
            factor = float(rope_params.get("factor", 1.0))
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (factor * pos_freqs)
            beta_fast = float(rope_params.get("beta_fast", 32))
            beta_slow = float(rope_params.get("beta_slow", 1))
            orig_max_pos = int(
                rope_params.get("original_max_position_embeddings", 4096)
            )
            low, high = _yarn_find_correction_range(
                beta_fast, beta_slow, head_dim, base, orig_max_pos
            )
            if low == high:
                high += 0.001
            ramp = torch.clamp(
                (torch.arange(half, dtype=torch.float32) - low) / (high - low),
                0,
                1,
            )
            extrapolation_factor = float(
                rope_params.get("extrapolation_factor", 1.0)
            )
            inv_freq_mask = (1 - ramp) * extrapolation_factor
            freqs = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
            attn_factor = float(rope_params.get("attn_factor", 1.0))
            mscale = (
                _yarn_get_mscale(factor, float(rope_params.get("mscale", 1.0)))
                / _yarn_get_mscale(
                    factor, float(rope_params.get("mscale_all_dim", 0.0))
                )
                * attn_factor
            )
        else:
            freqs = 1.0 / pos_freqs
            mscale = 1.0

        positions = torch.arange(max_pos, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # [max_pos, half]
        cos = angles.cos() * mscale
        sin = angles.sin() * mscale
        if interleaved:
            # GPT-J / DeepSeek-MLA: [c0,c0,c1,c1,...] so a kernel rotating
            # pairs (x[2i], x[2i+1]) reads the matching angle.
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        else:
            # HF/LLaMA cat-convention: duplicate the half across [:D/2],[D/2:].
            cos = torch.cat((cos, cos), dim=-1)
            sin = torch.cat((sin, sin), dim=-1)
        return cos.to(torch.bfloat16), sin.to(torch.bfloat16)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lookup ``(cos[positions], sin[positions])`` (bf16, same device)."""
        if not torch.is_tensor(positions):
            raise TypeError(
                "RotaryEmbedding.forward expects a torch.Tensor of "
                f"position indices; got {type(positions).__name__}."
            )
        return self.cos[positions], self.sin[positions]

    def auto_grid_dim(self, *args, **kwargs):
        """Not applicable — RotaryEmbedding emits no MPK task."""
        raise NotImplementedError(
            "RotaryEmbedding does not emit an MPK task; the RoPE rotation "
            "is performed inside the attention kernel that consumes "
            "(cos, sin) DTensors returned by RotaryEmbedding.compile()."
        )

    def compile(self) -> Tuple[DTensor, DTensor]:
        """Attach precomputed cos/sin buffers to the active PK (no task emitted).

        Tensor contract:
          cos_dt: (max_position_embeddings, head_dim) bf16, RoPE table.
          sin_dt: (max_position_embeddings, head_dim) bf16, RoPE table.

        Notes: returns the two DTensors threaded into
        ``pk.attention_layer(cos_pos_embed=..., sin_pos_embed=...)``. The
        actual RoPE rotation lives inside the attention kernel.
        """
        from ..context import current_pk

        pk = current_pk()
        cos_name = f"{self.prefix}cos" if self.prefix else "rotary_cos"
        sin_name = f"{self.prefix}sin" if self.prefix else "rotary_sin"
        cos_dt = pk.attach_input(self.cos, name=cos_name)
        sin_dt = pk.attach_input(self.sin, name=sin_name)
        return cos_dt, sin_dt
