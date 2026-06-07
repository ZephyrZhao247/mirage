"""V4-Flash ``flash_mla_with_kvcache`` (sparse decode) -- NEW Blackwell
naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/flash_mla_with_kvcache.md``.

Naive port of the closed-source FlashMLA sparse decode kernel. One CTA
per (token, head); walks indices one-at-a-time, gathers + dequants from
the paged uint8 cache, does QK + online softmax + PV + optional sink.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FlashMLADecode"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4FlashMLADecode(MPKModule):
    """V4-Flash sparse MLA decode (one task per (token, head))."""

    def __init__(
        self,
        num_heads_q: int = 64,
        head_dim: int = 576,
        head_v: int = 512,
        cache_block_size: int = 64,
        softmax_scale: Optional[float] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim != 576 or head_v != 512:
            raise ValueError(
                "V4FlashMLADecode locks head_dim=576, head_v=512"
            )
        if num_heads_q not in (64, 128):
            raise ValueError(
                f"num_heads_q must be in {{64, 128}}; got {num_heads_q}"
            )
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        self.head_v = head_v
        self.cache_block_size = cache_block_size
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        self.softmax_scale = float(softmax_scale)

    def forward(self, *args, **kwargs):
        """Reference is implemented inside the test (FP8 cache layout
        is layout-specific)."""
        raise NotImplementedError(
            "V4FlashMLADecode.forward(): write the reference in the test."
        )

    def auto_grid_dim(self, q_dt: DTensor) -> GridDim:
        """One CTA per (token, head)."""
        return (q_dt.dim(0), self.num_heads_q, 1)

    def compile(
        self,
        q: DTensor,
        k_cache: DTensor,
        indices: DTensor,
        topk_length: DTensor,
        attn_sink: DTensor,
        *,
        block_stride_bytes: int,
        head_id: int = 0,
        out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``flash_mla_decode_v4_sm100`` task per head.

        Tensor contract:
          q            : (T, NUM_HEADS_Q, 576) bf16.
          k_cache      : (num_blocks, block_stride_bytes) uint8.
          indices      : (T, topk) int32.
          topk_length  : (T,) int32.
          attn_sink    : (NUM_HEADS_Q,) fp32.
          out          : (T, NUM_HEADS_Q, 512) bf16.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q)
        if block_dim is None:
            block_dim = self.default_block_dim()

        T = q.dim(0)
        if out is None:
            out_dt = pk.new_tensor(
                dims=(T, self.num_heads_q, self.head_v),
                dtype=q.dtype,
                name=f"{self.prefix}flash_mla_decode_out",
            )
        elif isinstance(out, torch.Tensor):
            out_dt = pk.attach_input(out,
                                     name=f"{self.prefix}flash_mla_decode_out")
        elif isinstance(out, DTensor):
            out_dt = out
        else:
            raise TypeError("out must be None, Tensor, or DTensor")

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (0, -1, -1), 1, True)
        tb_graph.new_input(k_cache, (-1, -1, -1), 0, True)
        tb_graph.new_input(indices, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_length, (0, -1, -1), 1, True)
        tb_graph.new_input(attn_sink, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [q, k_cache, indices, topk_length, attn_sink, out_dt],
            tb_graph,
        )
        softmax_scale_x1e6 = int(round(self.softmax_scale * 1.0e6))
        pk.kn_graph.register_task(
            tb_graph,
            "flash_mla_decode_v4_sm100",
            [block_stride_bytes, self.cache_block_size, head_id,
             softmax_scale_x1e6],
        )
        return out_dt
