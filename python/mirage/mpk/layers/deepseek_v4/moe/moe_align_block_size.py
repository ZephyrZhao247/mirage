"""V4-Flash ``moe_align_block_size`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/moe_align_block_size.md``.

Decision: **NEW**.

Whole-batch bin-by-expert + pad-to-block_size dispatch.  vLLM's CUDA
implementation runs a 2-block histogram/cumsum + a sort kernel; the
naive port collapses both stages into a single CTA that processes
all tokens sequentially in shared memory.  ``num_experts`` and
``block_size`` are bound at codegen time.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/moe_align_block_size_v4_sm100.cuh``
* Task name: ``moe_align_block_size_v4_sm100``
* Enum slot: ``TASK_MOE_ALIGN_BLOCK_SIZE_V4_SM100 = 388``

Per the user's directive, the kernel is implemented as a single-CTA
"all tokens" task: ``grid_dim = (1, 1, 1)``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import BlockDim, GridDim, MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4MoeAlignBlockSize"]


class V4MoeAlignBlockSize(MPKModule):
    """Whole-batch bin-by-expert.

    Constructor args:
      * ``num_tokens``      -- T.
      * ``top_k``           -- TOP_K.
      * ``num_experts``     -- E (global expert count).
      * ``block_size``      -- BLOCK_SIZE_M of the downstream GEMM.

    Outputs (allocated by the caller and passed to ``compile``):
      * ``sorted_token_ids``    int32 [EM]
      * ``expert_ids``          int32 [num_m_blocks]
      * ``num_tokens_post_pad`` int32 [1]
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        *,
        block_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_experts <= 0 or num_experts > 512:
            raise ValueError(
                f"V4MoeAlignBlockSize: num_experts out of supported range "
                f"(0, 512]; got {num_experts}."
            )
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size
        em = num_tokens * top_k + num_experts * (block_size - 1)
        em = ((em + block_size - 1) // block_size) * block_size
        self.em = em
        self.num_m_blocks = em // block_size

    def forward(
        self, topk_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Faithful PyTorch reference of the alignment op."""
        T, K = topk_ids.shape
        assert T == self.num_tokens and K == self.top_k
        E = self.num_experts
        bs = self.block_size
        ids_flat = topk_ids.reshape(-1).to(torch.long).cpu()
        numel = ids_flat.numel()

        counts = torch.bincount(ids_flat, minlength=E).tolist()
        padded = [((c + bs - 1) // bs) * bs for c in counts]
        cumsum = [0] * (E + 1)
        for e in range(E):
            cumsum[e + 1] = cumsum[e] + padded[e]
        num_tokens_post_pad = cumsum[E]

        sorted_token_ids = torch.full(
            (self.em,), numel, dtype=torch.int32
        )
        per_expert_offset = cumsum[:E]
        offset = list(per_expert_offset)
        for i in range(numel):
            e = int(ids_flat[i].item())
            if 0 <= e < E:
                slot = offset[e]
                if slot < self.em:
                    sorted_token_ids[slot] = i
                    offset[e] += 1

        expert_ids = torch.full((self.num_m_blocks,), -1, dtype=torch.int32)
        last_block = num_tokens_post_pad // bs
        for b in range(last_block):
            target = b * bs
            for e in range(E):
                if cumsum[e] <= target < cumsum[e + 1]:
                    expert_ids[b] = e
                    break

        num_post = torch.tensor([num_tokens_post_pad], dtype=torch.int32)
        device = topk_ids.device
        return (
            sorted_token_ids.to(device),
            expert_ids.to(device),
            num_post.to(device),
        )

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        # Single-CTA whole-batch view, per user directive.
        return (1, 1, 1)

    def compile(
        self,
        topk_ids: DTensor,
        sorted_token_ids: DTensor,
        expert_ids: DTensor,
        num_tokens_post_pad: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(topk_ids)
        block_dim = block_dim or self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(topk_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(sorted_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_tokens_post_pad, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [topk_ids, sorted_token_ids, expert_ids, num_tokens_post_pad],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "moe_align_block_size_v4_sm100",
            [self.num_experts, self.block_size],
        )
        return sorted_token_ids, expert_ids, num_tokens_post_pad
