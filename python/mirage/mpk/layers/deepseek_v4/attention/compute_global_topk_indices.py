"""V4-Flash ``compute_global_topk_indices_and_lens`` -- NEW Blackwell naive.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/compute_global_topk_indices_and_lens.md``.

Decision: **NEW**.

Per-token block-table indirection + valid-count for C4A decode:
  for each token t:
    for each topk lane k:
      local = topk_indices[t, k]
      if local < 0: global = -1
      else:         global = block_table[req, local // BS] * BS + local % BS; count++
    topk_lens[t] = is_valid_token[t] ? count : 0

Naive design
------------
* One CTA per query token (``grid = (num_tokens, 1, 1)``).
* 256-thread CTA strides across the ``topk`` lanes; warp/block reduce
  combines the per-thread valid counts. The whole topk row is read once.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4ComputeGlobalTopkIndices"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4ComputeGlobalTopkIndices(MPKModule):
    """Naive C4A topk index globalizer.

    Constructor args:
        topk: per-token topk width (V4-Flash: 2048).
        block_size: page block size in the *compressed* cache divided
            by compress_ratio (V4-Flash C4A: 256/4 = 64).
    """

    def __init__(
        self,
        topk: int,
        block_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if topk <= 0:
            raise ValueError(f"topk must be positive; got {topk}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive; got {block_size}")
        self.topk = int(topk)
        self.block_size = int(block_size)

    # ------------------------------------------------------------------
    def forward(
        self,
        topk_indices: torch.Tensor,
        token_to_req: torch.Tensor,
        block_table: torch.Tensor,
        is_valid_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference matching the Triton kernel."""
        assert topk_indices.dim() == 2 and topk_indices.shape[1] == self.topk
        assert topk_indices.dtype == torch.int32
        assert token_to_req.dtype == torch.int32
        assert block_table.dtype == torch.int32 and block_table.dim() == 2

        T = topk_indices.shape[0]
        global_topk = torch.empty_like(topk_indices)
        topk_lens = torch.empty(T, dtype=torch.int32, device=topk_indices.device)

        for t in range(T):
            r = int(token_to_req[t].item())
            is_valid = bool(is_valid_token[t].item() != 0)
            count = 0
            for k in range(self.topk):
                local = int(topk_indices[t, k].item())
                if local < 0:
                    global_topk[t, k] = -1
                else:
                    block_in_seq = local // self.block_size
                    off = local % self.block_size
                    phys = int(block_table[r, block_in_seq].item())
                    global_topk[t, k] = phys * self.block_size + off
                    count += 1
            topk_lens[t] = count if is_valid else 0
        return global_topk, topk_lens

    # ------------------------------------------------------------------
    def auto_grid_dim(self, topk_indices_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = topk_indices_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    def compile(
        self,
        topk_indices: DTensor,
        token_to_req: DTensor,
        block_table: DTensor,
        is_valid_token: DTensor,
        *,
        global_topk_indices: Optional[DTensor] = None,
        topk_lens: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        from .....core import CyTBGraph, int32 as _mi_i32
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(topk_indices)
        if block_dim is None:
            block_dim = self.default_block_dim()

        max_blocks_per_seq = block_table.dim(1)
        T = topk_indices.dim(0)

        if global_topk_indices is None:
            global_topk_indices = pk.new_tensor(
                dims=(T, self.topk),
                dtype=_mi_i32,
                name=f"{self.prefix}global_topk_indices",
            )
        if topk_lens is None:
            topk_lens = pk.new_tensor(
                dims=(T,),
                dtype=_mi_i32,
                name=f"{self.prefix}topk_lens",
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(topk_indices, (0, -1, -1), 1, True)
        tb_graph.new_input(token_to_req, (0, -1, -1), 1, True)
        tb_graph.new_input(block_table, (-1, -1, -1), 0, True)
        tb_graph.new_input(is_valid_token, (0, -1, -1), 1, True)
        tb_graph.new_input(global_topk_indices, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_lens, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [topk_indices, token_to_req, block_table, is_valid_token,
             global_topk_indices, topk_lens], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "compute_global_topk_indices_v4_sm100",
            [self.topk, self.block_size, max_blocks_per_seq],
        )
        return global_topk_indices, topk_lens
