"""V4-Flash ``write_zeros_to_output`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/write_zeros_to_output.md``.

Decision: **NEW**.

The vLLM helper is an inlined device function called from inside
``fused_moe_kernel`` when ``expert_ids[pid_m] == -1``.  Here we expose
it as a standalone MPK task: a per-(pid_m, pid_n) C-tile zero-fill
gated by ``token_mask``.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/write_zeros_to_output_v4_sm100.cuh``
* Task name: ``write_zeros_to_output_v4_sm100``
* Enum slot: ``TASK_WRITE_ZEROS_TO_OUTPUT_V4_SM100 = 387``

Naive impl: one CTA per (pid_m, pid_n) tile; ``block = (256, 1, 1)``.
The grid is ``(num_m_blocks, num_n_blocks, 1)``; the runtime stores
``pid_m`` in ``task_metadata.request_id`` and ``pid_n`` in
``task_metadata.kv_idx``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import BlockDim, GridDim, MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4WriteZerosToOutput"]


class V4WriteZerosToOutput(MPKModule):
    """Zero-fill C-tiles for `(expert_ids[pid_m] == -1)` blocks.

    Constructor args:
      * ``num_tokens``       -- T (max tokens in the batch).
      * ``top_k``            -- TOP_K (routing slots per token).
      * ``output_dim``       -- N dimension of C ([T, top_k, N]).
      * ``block_m``          -- BLOCK_SIZE_M.
      * ``block_n``          -- BLOCK_SIZE_N.
      * ``num_experts``      -- E.  Used for shape derivation of the EM
                                (``num_tokens * top_k + num_experts *
                                (block_m - 1)`` upper bound).
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        output_dim: int,
        *,
        block_m: int,
        block_n: int,
        num_experts: int,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.output_dim = output_dim
        self.block_m = block_m
        self.block_n = block_n
        self.num_experts = num_experts

        # EM upper-bound, mirroring vLLM's moe_align_block_size:
        em = num_tokens * top_k + num_experts * (block_m - 1)
        # Round up to a multiple of block_m.
        em = ((em + block_m - 1) // block_m) * block_m
        self.em = em
        self.num_m_blocks = em // block_m
        self.num_n_blocks = (output_dim + block_n - 1) // block_n

    def forward(
        self,
        c: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ) -> torch.Tensor:
        """Faithful reference: zero rows in C corresponding to -1-expert blocks."""
        T = self.num_tokens
        top_k = self.top_k
        N = self.output_dim
        block_m = self.block_m
        num_valid = T * top_k
        num_post = int(num_tokens_post_pad.item())
        out = c.clone()
        for pid_m in range(self.num_m_blocks):
            if pid_m * block_m >= num_post:
                break
            if int(expert_ids[pid_m].item()) != -1:
                continue
            offs = sorted_token_ids[pid_m * block_m:(pid_m + 1) * block_m]
            for m in range(min(block_m, offs.numel())):
                ot = int(offs[m].item())
                if ot < num_valid:
                    out.view(T * top_k, N)[ot, :] = 0
        return out

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        pk = current_pk()
        total = self.num_m_blocks * self.num_n_blocks
        # We need (num_m_blocks, num_n_blocks, 1) so request_id=bid.x and
        # kv_idx=bid.y can carry pid_m and pid_n.  Cap total by num_workers.
        gx = self.num_m_blocks
        gy = self.num_n_blocks
        if total > pk.num_workers:
            # Best-effort cap: prefer reducing the N axis first.
            gy = max(1, min(gy, pk.num_workers // max(gx, 1)))
        return (max(1, gx), max(1, gy), 1)

    def compile(
        self,
        c: DTensor,
        sorted_token_ids: DTensor,
        expert_ids: DTensor,
        num_tokens_post_pad: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(c)
        block_dim = block_dim or self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(sorted_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_tokens_post_pad, (-1, -1, -1), -1, True)
        tb_graph.new_input(c, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [sorted_token_ids, expert_ids, num_tokens_post_pad, c], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph,
            "write_zeros_to_output_v4_sm100",
            [self.block_m, self.block_n],
        )
        return c
