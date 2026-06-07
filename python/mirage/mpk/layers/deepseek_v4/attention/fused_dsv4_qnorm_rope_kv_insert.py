"""V4-Flash ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`` --
NEW Blackwell naive kernel (THE BIG ONE).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md``.

5-op fusion per token: Q head-norm (no weight) + Q RoPE + KV RoPE +
UE8M0 FP8 block quant of KV[..., 0:448] + scatter into paged uint8
cache at slot_mapping[t].
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedDSV4QNormRopeKVInsert"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

HEAD_DIM = 512
ROPE_DIM = 64
QUANT_BLOCK = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
NUM_QUANT_BLOCKS = NOPE_DIM // QUANT_BLOCK
SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1
TOKEN_BYTES = NOPE_DIM + ROPE_DIM * 2


class V4FusedDSV4QNormRopeKVInsert(MPKModule):
    """V4-Flash 5-op monolithic Q-norm + RoPE + KV-quant + cache-insert.

    Constructor args:

      * ``num_heads_q``      -- live Q heads (e.g. 64).
      * ``q_head_padded``    -- padded Q heads in {8,16,32,64,128}.
      * ``cache_block_size`` -- tokens per paged-cache block (e.g. 64).
      * ``num_blocks``       -- number of paged blocks.
      * ``eps``              -- affects ``forward()`` only; kernel
                                hard-codes ``1e-6f``.
    """

    def __init__(
        self,
        num_heads_q: int,
        q_head_padded: int,
        cache_block_size: int,
        num_blocks: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if q_head_padded < num_heads_q:
            raise ValueError(
                f"q_head_padded ({q_head_padded}) < num_heads_q "
                f"({num_heads_q})"
            )
        self.num_heads_q = num_heads_q
        self.q_head_padded = q_head_padded
        self.cache_block_size = cache_block_size
        self.num_blocks = num_blocks
        self.block_stride_bytes = cache_block_size * (TOKEN_BYTES +
                                                       SCALE_BYTES_PER_TOKEN)
        self.eps = eps

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference. Returns ``(q_out, k_cache)``."""
        T = q_in.shape[0]
        device = q_in.device
        dtype = q_in.dtype
        q_out = torch.zeros(
            T, self.q_head_padded, HEAD_DIM, device=device, dtype=dtype
        )
        if k_cache is None:
            k_cache = torch.zeros(
                self.num_blocks, self.block_stride_bytes,
                device=device, dtype=torch.uint8,
            )

        q_f = q_in.to(torch.float32)
        rrms = torch.rsqrt(q_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        q_f = q_f * rrms

        cos = cos_sin_cache[positions, :ROPE_DIM // 2]
        sin = cos_sin_cache[positions, ROPE_DIM // 2:]
        q_rope = q_f[..., NOPE_DIM:].reshape(T, self.num_heads_q,
                                              ROPE_DIM // 2, 2)
        even = q_rope[..., 0]
        odd = q_rope[..., 1]
        c = cos[:, None, :]
        s = sin[:, None, :]
        new_even = even * c - odd * s
        new_odd = odd * c + even * s
        q_rope_new = torch.stack([new_even, new_odd], dim=-1).reshape(
            T, self.num_heads_q, ROPE_DIM
        )
        q_f[..., NOPE_DIM:] = q_rope_new
        q_out[:, :self.num_heads_q, :] = q_f.to(dtype)

        kv_f = kv_in.to(torch.float32)
        kv_rope = kv_f[..., NOPE_DIM:].reshape(T, ROPE_DIM // 2, 2)
        even = kv_rope[..., 0]
        odd = kv_rope[..., 1]
        new_even = even * cos - odd * sin
        new_odd = odd * cos + even * sin
        kv_rope_new = torch.stack([new_even, new_odd], dim=-1).reshape(
            T, ROPE_DIM
        )
        kv_f[..., NOPE_DIM:] = kv_rope_new

        for t in range(T):
            slot_id = int(slot_mapping[t].item())
            if slot_id < 0:
                continue
            block_idx = slot_id // self.cache_block_size
            pos_in_block = slot_id % self.cache_block_size
            base = pos_in_block * TOKEN_BYTES
            for qb in range(NUM_QUANT_BLOCKS):
                lo = qb * QUANT_BLOCK
                hi = lo + QUANT_BLOCK
                chunk = kv_f[t, lo:hi]
                absmax = chunk.abs().amax().clamp_min(1e-4)
                exponent = int(torch.ceil(torch.log2(absmax / 448.0)).item())
                inv_scale = 2.0 ** (-exponent)
                q = (chunk * inv_scale).clamp(-448.0, 448.0)
                q_fp8 = q.to(torch.float8_e4m3fn)
                k_cache[block_idx, base + lo:base + hi] = q_fp8.view(
                    torch.uint8
                )
                scale_base = (self.cache_block_size * TOKEN_BYTES +
                              pos_in_block * SCALE_BYTES_PER_TOKEN)
                biased = max(0, min(255, exponent + 127))
                k_cache[block_idx, scale_base + qb] = biased
            scale_base = (self.cache_block_size * TOKEN_BYTES +
                          pos_in_block * SCALE_BYTES_PER_TOKEN)
            k_cache[block_idx, scale_base + NUM_QUANT_BLOCKS] = 0
            rope_bf16 = kv_f[t, NOPE_DIM:].to(torch.bfloat16)
            k_cache[block_idx, base + NOPE_DIM:base + TOKEN_BYTES] = (
                rope_bf16.view(torch.uint8)
            )

        return q_out, k_cache

    def auto_grid_dim(self, q_in_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``)."""
        pk = current_pk()
        return (max(1, min(q_in_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        q_in: DTensor,
        kv_in: DTensor,
        slot_mapping: DTensor,
        positions: DTensor,
        cos_sin_cache: DTensor,
        k_cache: DTensor,
        *,
        q_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``fused_dsv4_qnorm_rope_kv_insert_v4_sm100`` task.

        Tensor contract:
          q_in           : (T, NUM_HEADS_Q, 512)  bf16, row-major.
          kv_in          : (T, 512)               bf16.
          slot_mapping   : (T,)                   int64.
          positions      : (T,)                   int64.
          cos_sin_cache  : (max_pos, 64)          fp32 broadcast.
          k_cache        : (num_blocks, block_stride_bytes) uint8 (out).
          q_out          : (T, Q_HEAD_PADDED, 512) bf16, allocated if None.

        Notes: block_stride_bytes = cache_block_size * 584.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_in)
        if block_dim is None:
            block_dim = self.default_block_dim()

        T = q_in.dim(0)
        if q_out is None:
            q_out_dt = pk.new_tensor(
                dims=(T, self.q_head_padded, HEAD_DIM),
                dtype=q_in.dtype,
                name=f"{self.prefix}q_out",
            )
        elif isinstance(q_out, torch.Tensor):
            q_out_dt = pk.attach_input(q_out, name=f"{self.prefix}q_out")
        elif isinstance(q_out, DTensor):
            q_out_dt = q_out
        else:
            raise TypeError("q_out must be None, Tensor, or DTensor")

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_in, (0, -1, -1), 1, True)
        tb_graph.new_input(kv_in, (0, -1, -1), 1, True)
        tb_graph.new_input(slot_mapping, (0, -1, -1), 1, True)
        tb_graph.new_input(positions, (0, -1, -1), 1, True)
        tb_graph.new_input(cos_sin_cache, (-1, -1, -1), 0, True)
        tb_graph.new_input(q_out_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), 0, True)
        pk.kn_graph.customized(
            [q_in, kv_in, slot_mapping, positions, cos_sin_cache,
             q_out_dt, k_cache],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_dsv4_qnorm_rope_kv_insert_v4_sm100",
            [self.block_stride_bytes, self.cache_block_size],
        )
        return q_out_dt, k_cache
