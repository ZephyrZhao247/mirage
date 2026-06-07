"""V4-Flash ``combine_topk_swa_indices`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/combine_topk_swa_indices.md``.

Decision: **NEW**.

Concatenate per-token compressed-pool topk indices with the per-token
SWA window indices into one flat per-token index row. Output is padded
to ``combined_topk = align_up(top_k + window_size, 128)`` with ``-1``
(the catalog pre-fills the output buffer; the kernel writes only the
valid prefix).

Naive design
------------
* One CTA per query token.
* Threads stripe across the topk and SWA portions.
* `token_to_batch`, `positions`, `gather_start` are precomputed by the
  caller from `query_start_loc`, `seq_lens`, `gather_lens` (cheaper to
  compute on CPU/Python than to re-derive in-kernel for the naive port).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4CombineTopkSwaIndices"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def combined_topk_for(top_k: int, window_size: int) -> int:
    """Return next-multiple-of-128 of (top_k + window_size)."""
    total = top_k + window_size
    return (total + SPARSE_PREFILL_TOPK_ALIGNMENT - 1) // SPARSE_PREFILL_TOPK_ALIGNMENT * SPARSE_PREFILL_TOPK_ALIGNMENT


class V4CombineTopkSwaIndices(MPKModule):
    """Naive prefill topk+SWA index combiner.

    Constructor args:
        top_k: per-token compressed-pool topk width (V4-Flash C4A: 2048;
            SWA-only layer passes 0).
        compress_ratio: 4 (C4A), 128 (C128A), or 1 (SWA-only placeholder).
        window_size: SWA window length (V4-Flash: 128).
        M: per-request stride in the chunk-flat KV workspace.
        N: per-request offset of the SWA region within M.
    """

    def __init__(
        self,
        top_k: int,
        compress_ratio: int,
        window_size: int,
        M: int,
        N: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if window_size <= 0:
            raise ValueError(f"window_size must be positive; got {window_size}")
        if M <= 0 or N < 0:
            raise ValueError(f"need M>0, N>=0; got M={M}, N={N}")
        self.top_k = int(top_k)
        self.compress_ratio = int(compress_ratio)
        self.window_size = int(window_size)
        self.M = int(M)
        self.N = int(N)
        self.combined_topk = combined_topk_for(self.top_k, self.window_size)

    # ------------------------------------------------------------------
    def forward(
        self,
        topk_indices: torch.Tensor,
        token_to_batch: torch.Tensor,
        positions: torch.Tensor,
        gather_start: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference."""
        assert topk_indices.dtype == torch.int32
        assert token_to_batch.dtype == torch.int32
        assert positions.dtype == torch.int32
        assert gather_start.dtype == torch.int32

        T = topk_indices.shape[0]
        combined = torch.full(
            (T, self.combined_topk), -1,
            dtype=torch.int32, device=topk_indices.device,
        )
        combined_lens = torch.empty(T, dtype=torch.int32, device=topk_indices.device)

        for t in range(T):
            batch = int(token_to_batch[t].item())
            pos = int(positions[t].item())
            gs = int(gather_start[t].item())

            topk_len = 0
            if self.top_k > 0 and self.compress_ratio > 0:
                topk_len = min((pos + 1) // self.compress_ratio, self.top_k)
            swa_len = max(0, min(pos + 1, self.window_size))

            for k in range(topk_len):
                local = int(topk_indices[t, k].item())
                combined[t, k] = -1 if local < 0 else local + self.M * batch

            for k in range(swa_len):
                combined[t, topk_len + k] = (
                    self.M * batch + self.N + (k + pos - swa_len + 1 - gs)
                )
            combined_lens[t] = topk_len + swa_len

        return combined, combined_lens

    # ------------------------------------------------------------------
    def auto_grid_dim(self, topk_indices_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = topk_indices_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    def compile(
        self,
        topk_indices: DTensor,
        token_to_batch: DTensor,
        positions: DTensor,
        gather_start: DTensor,
        combined_indices: DTensor,
        *,
        combined_lens: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``combine_topk_swa_indices_v4_sm100`` task.

        Tensor contract:
          topk_indices     : (T, top_k)       int32.
          token_to_batch   : (T,)             int32.
          positions        : (T,)             int32 (absolute seq positions).
          gather_start     : (T,)             int32 (per-token gather_start).
          combined_indices : (T, combined_topk) int32 -- caller pre-fills -1.
        """
        from .....core import CyTBGraph, int32 as _mi_i32
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(topk_indices)
        if block_dim is None:
            block_dim = self.default_block_dim()

        T = topk_indices.dim(0)
        if combined_lens is None:
            combined_lens = pk.new_tensor(
                dims=(T,), dtype=_mi_i32, name=f"{self.prefix}combined_lens"
            )

        if combined_indices.dim(1) != self.combined_topk:
            raise ValueError(
                f"combined_indices last-dim ({combined_indices.dim(1)}) != "
                f"combined_topk ({self.combined_topk})"
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(topk_indices, (0, -1, -1), 1, True)
        tb_graph.new_input(token_to_batch, (0, -1, -1), 1, True)
        tb_graph.new_input(positions, (0, -1, -1), 1, True)
        tb_graph.new_input(gather_start, (0, -1, -1), 1, True)
        tb_graph.new_input(combined_indices, (0, -1, -1), 1, True)
        tb_graph.new_input(combined_lens, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [topk_indices, token_to_batch, positions, gather_start,
             combined_indices, combined_lens], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "combine_topk_swa_indices_v4_sm100",
            [self.top_k, self.compress_ratio, self.window_size,
             self.M, self.N, self.combined_topk],
        )
        return combined_indices, combined_lens
