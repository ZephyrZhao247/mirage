"""V4-Flash ``fused_inv_rope_fp8_quant`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_inv_rope_fp8_quant.md``.

Inverse GPT-J RoPE + UE8M0 block-FP8 quant per (token, head). Output:
o_fp8 (uint8 = e4m3 raw bytes) and o_scale (INT32 UE8M0-packed). The
INT32 scale dtype is the sm_100a contract; downstream
``deepseek_v4_fp8_einsum`` consumes exactly this layout.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedInvRopeFP8Quant"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

HEAD_DIM = 512
ROPE_DIM = 64
QUANT_GROUP = 128
CHUNKS_PER_HEAD = HEAD_DIM // QUANT_GROUP
FP8_MAX = 448.0


class V4FusedInvRopeFP8Quant(MPKModule):
    """V4-Flash inverse-RoPE + UE8M0 block-FP8 quant.

    Constructor args:

      * ``n_groups``        -- ``o_groups`` (V4-Flash: 8).
      * ``heads_per_group`` -- ``num_heads // n_groups`` (V4-Flash: 8).
      * ``head_dim``        -- 512 locked.
      * ``rope_dim``        -- 64 locked.
      * ``quant_group``     -- 128 locked.
    """

    def __init__(
        self,
        n_groups: int = 8,
        heads_per_group: int = 8,
        head_dim: int = HEAD_DIM,
        rope_dim: int = ROPE_DIM,
        quant_group: int = QUANT_GROUP,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim != HEAD_DIM or rope_dim != ROPE_DIM or \
                quant_group != QUANT_GROUP:
            raise ValueError(
                "V4FusedInvRopeFP8Quant locks HEAD_DIM=512, ROPE_DIM=64, "
                "QUANT_GROUP=128"
            )
        self.n_groups = n_groups
        self.heads_per_group = heads_per_group
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.quant_group = quant_group
        self.num_heads = n_groups * heads_per_group
        bytes_per_group = CHUNKS_PER_HEAD * heads_per_group
        self.scale_inner = (bytes_per_group + 3) // 4

    def forward(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference matching the per-(token, head) pipeline.

        Returns ``(o_fp8, o_scale)``:
          o_fp8   : (T, n_groups, heads_per_group * head_dim) uint8
          o_scale : (T, n_groups, scale_inner) int32
        """
        T, H, D = o.shape
        device = o.device

        cos = cos_sin_cache[positions, :ROPE_DIM // 2]
        sin = cos_sin_cache[positions, ROPE_DIM // 2:]
        o_f = o.to(torch.float32).clone()
        nope = D - ROPE_DIM
        rope_pairs = o_f[:, :, nope:].reshape(T, H, ROPE_DIM // 2, 2)
        e = rope_pairs[..., 0]
        d = rope_pairs[..., 1]
        c = cos[:, None, :]
        s = sin[:, None, :]
        new_even = e * c + d * s
        new_odd = -e * s + d * c
        rope_inv = torch.stack([new_even, new_odd], dim=-1).reshape(
            T, H, ROPE_DIM
        )
        o_f[:, :, nope:] = rope_inv

        chunks = o_f.reshape(T, H, CHUNKS_PER_HEAD, QUANT_GROUP)
        absmax = chunks.abs().amax(dim=-1).clamp_min(1e-10)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        inv_scale = torch.exp2(-exponent)
        scaled = (chunks * inv_scale[..., None]).clamp(-FP8_MAX, FP8_MAX)
        o_fp8_flat = scaled.to(torch.float8_e4m3fn)
        o_fp8 = o_fp8_flat.reshape(
            T, self.n_groups, self.heads_per_group, D
        ).reshape(
            T, self.n_groups, self.heads_per_group * D
        ).view(torch.uint8)

        biased = (exponent + 127.0).clamp(0.0, 255.0).to(torch.int32)
        biased = biased.reshape(T, self.n_groups, self.heads_per_group,
                                CHUNKS_PER_HEAD)
        bytes_per_group = CHUNKS_PER_HEAD * self.heads_per_group
        flat = biased.reshape(T, self.n_groups, bytes_per_group)
        pad = (4 - bytes_per_group % 4) % 4
        if pad > 0:
            flat = torch.cat(
                [
                    flat,
                    torch.zeros(T, self.n_groups, pad, dtype=torch.int32,
                                device=device),
                ],
                dim=-1,
            )
        packed = flat.reshape(T, self.n_groups, -1, 4)
        weights = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24],
                                dtype=torch.int32, device=device)
        o_scale = (packed * weights).sum(dim=-1).to(torch.int32)
        return o_fp8, o_scale

    def auto_grid_dim(self, o_dt: DTensor) -> GridDim:
        """One CTA per (token, head)."""
        return (o_dt.dim(0), self.num_heads, 1)

    def default_block_dim(self) -> BlockDim:
        """Single-warp kernel (matches Triton num_warps=1)."""
        return (32, 1, 1)

    def compile(
        self,
        o: DTensor,
        positions: DTensor,
        cos_sin_cache: DTensor,
        *,
        head_id: int = 0,
        o_fp8: Optional[torch.Tensor] = None,
        o_scale: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``fused_inv_rope_fp8_quant_v4_sm100`` task per head.

        Tensor contract:
          o              : (T, NUM_HEADS, 512) bf16.
          positions      : (T,)                int64.
          cos_sin_cache  : (max_pos, 64)       fp32 broadcast.
          o_fp8          : (T, n_groups, heads_per_group*512) uint8.
          o_scale        : (T, n_groups, scale_inner)         int32.

        Notes: ``o_scale`` writes via atomicOr (single-warp, low contention).
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(o)
        if block_dim is None:
            block_dim = self.default_block_dim()

        T = o.dim(0)
        if o_fp8 is None:
            o_fp8_dt = pk.new_tensor(
                dims=(T, self.n_groups,
                      self.heads_per_group * self.head_dim),
                dtype=torch.uint8,
                name=f"{self.prefix}o_fp8",
            )
        elif isinstance(o_fp8, torch.Tensor):
            o_fp8_dt = pk.attach_input(o_fp8, name=f"{self.prefix}o_fp8")
        elif isinstance(o_fp8, DTensor):
            o_fp8_dt = o_fp8
        else:
            raise TypeError("o_fp8 must be None, Tensor, or DTensor")

        if o_scale is None:
            o_scale_dt = pk.new_tensor(
                dims=(T, self.n_groups, self.scale_inner),
                dtype=torch.int32,
                name=f"{self.prefix}o_scale",
            )
        elif isinstance(o_scale, torch.Tensor):
            o_scale_dt = pk.attach_input(o_scale,
                                         name=f"{self.prefix}o_scale")
        elif isinstance(o_scale, DTensor):
            o_scale_dt = o_scale
        else:
            raise TypeError("o_scale must be None, Tensor, or DTensor")

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(o, (0, -1, -1), 1, True)
        tb_graph.new_input(positions, (0, -1, -1), 1, True)
        tb_graph.new_input(cos_sin_cache, (-1, -1, -1), 0, True)
        tb_graph.new_input(o_fp8_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(o_scale_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [o, positions, cos_sin_cache, o_fp8_dt, o_scale_dt], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_inv_rope_fp8_quant_v4_sm100",
            [head_id],
        )
        return o_fp8_dt, o_scale_dt
